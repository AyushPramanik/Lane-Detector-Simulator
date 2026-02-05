#include "LaneDetector.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

LaneDetector::LaneDetector(const LaneConfig& config) : config_(config) {}

void LaneDetector::setConfig(const LaneConfig& config) { config_ = config; }

const LaneConfig& LaneDetector::getConfig() const noexcept { return config_; }

cv::Mat LaneDetector::processFrame(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::invalid_argument("LaneDetector::processFrame received an empty frame");
    }

    const int h = frame.rows;
    const int w = frame.cols;

    // 1. Grayscale + blur
    cv::Mat blurred = preprocessFrame(frame);

    // 2. Canny edge detection
    cv::Mat edges = detectEdges(blurred);

    // 3. Trapezoidal ROI mask
    auto roi = computeROIVertices(w, h);
    cv::Mat maskedEdges = applyROI(edges, roi);

    // 4. Probabilistic Hough transform
    auto lines = detectLines(maskedEdges);

    // 5. Separate left/right, average, extrapolate
    auto [leftLane, rightLane] = computeLanes(lines, w, h);

    // 6. Blend overlay onto original frame
    return overlayLanes(frame, leftLane, rightLane);
}

// Stage 1: preprocessFrame

cv::Mat LaneDetector::preprocessFrame(const cv::Mat& frame) const {
    cv::Mat gray, blurred;

    // Grayscale — luminance information is sufficient for edge detection;
    // dropping two colour channels speeds up all subsequent operations.
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Gaussian blur — convolves with a Gaussian kernel to suppress high-
    // frequency noise before Canny computes image gradients.  A larger
    // kernel removes more noise but also blurs true edges; 5×5 is a good
    // default for HD footage.
    const int k = config_.gaussianKernelSize | 1; // ensure odd
    cv::GaussianBlur(gray, blurred, cv::Size(k, k), 0);

    return blurred;
}

// Stage 2: detectEdges

cv::Mat LaneDetector::detectEdges(const cv::Mat& blurred) const {
    cv::Mat edges;

    // Canny edge detector:
    //   1. Computes gradient magnitude (Sobel).
    //   2. Non-maximum suppression — thins edges to 1-pixel width.
    //   3. Hysteresis thresholding:
    //      * Pixels with gradient > cannyHigh  → definite edge.
    //      * Pixels with gradient < cannyLow   → suppressed.
    //      * Pixels in between                 → kept only if connected to a
    //                                            strong edge.
    cv::Canny(blurred, edges, config_.cannyLow, config_.cannyHigh);

    return edges;
}

// Stage 3: computeROIVertices + applyROI

std::vector<cv::Point> LaneDetector::computeROIVertices(int width, int height) const {

    const int bottomY     = height;
    const int topY        = static_cast<int>(height * config_.roiHeightRatio);
    const int cx          = width / 2;
    const int bottomHalf  = static_cast<int>(width * config_.roiBottomWidthRatio / 2.0);
    const int topHalf     = static_cast<int>(width * config_.roiTopWidthRatio  / 2.0);

    return {
        cv::Point(cx - bottomHalf, bottomY),  // BL
        cv::Point(cx - topHalf,    topY),      // TL
        cv::Point(cx + topHalf,    topY),      // TR
        cv::Point(cx + bottomHalf, bottomY)   // BR
    };
}

cv::Mat LaneDetector::applyROI(const cv::Mat& edges,
                                const std::vector<cv::Point>& roi) const {
    // Build a single-channel mask the same size as `edges`
    cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8UC1);

    // cv::fillPoly requires a pointer-to-array-of-points; wrap our vector
    const cv::Point* pts  = roi.data();
    const int        npts = static_cast<int>(roi.size());
    cv::fillPoly(mask, &pts, &npts, 1, cv::Scalar(255));

    // Bitwise-AND: keep only edge pixels that fall inside the polygon
    cv::Mat masked;
    cv::bitwise_and(edges, mask, masked);
    return masked;
}

// Stage 4: detectLines (Probabilistic Hough)

std::vector<cv::Vec4i> LaneDetector::detectLines(const cv::Mat& maskedEdges) const {
    std::vector<cv::Vec4i> lines;

    // HoughLinesP — probabilistic variant:
    //   * Faster than standard HoughLines because it works on a random subset.
    //   * Returns line *segments* (x1,y1,x2,y2) rather than infinite lines.
    //
    // Parameters:
    //   rho         — accumulator resolution (px); 1 is standard.
    //   theta       — angle resolution (rad); π/180 = 1°.
    //   threshold   — minimum votes a cell needs to be considered a line.
    //   minLength   — reject segments shorter than this.
    //   maxGap      — bridge collinear segments separated by ≤ this gap.
    cv::HoughLinesP(maskedEdges,
                    lines,
                    /*rho=*/   1,
                    /*theta=*/ CV_PI / 180.0,
                    config_.houghThreshold,
                    config_.houghMinLineLength,
                    config_.houghMaxLineGap);

    return lines;
}

// Stage 5: computeLanes — classify, average, extrapolate

std::pair<Lane, Lane> LaneDetector::computeLanes(
        const std::vector<cv::Vec4i>& lines,
        int width, int height) const
{
    std::vector<std::pair<double, double>> leftSI,  // (slope, intercept) for left lane
                                           rightSI; // … for right lane

    for (const auto& seg : lines) {
        const int x1 = seg[0], y1 = seg[1];
        const int x2 = seg[2], y2 = seg[3];

        if (x1 == x2) continue; // vertical segment → undefined slope

        const double slope     = static_cast<double>(y2 - y1) / (x2 - x1);
        const double intercept = y1 - slope * x1;

        // Discard near-horizontal lines — road lane markings always have a
        // noticeable slope when viewed from a forward-facing camera.
        if (std::abs(slope) < config_.minSlopeThreshold) continue;

        // In image space y increases downward:
        //   Left lane  → runs from bottom-left toward centre → negative slope.
        //   Right lane → runs from bottom-right toward centre → positive slope.
        //
        // We also sanity-check that the intercept keeps the line on-screen.
        if (slope < 0 && x1 < width && x2 < width) {
            leftSI.emplace_back(slope, intercept);
        } else if (slope > 0 && x1 >= 0 && x2 >= 0) {
            rightSI.emplace_back(slope, intercept);
        }
    }

    Lane leftLane  = fitLane(leftSI,  height);
    Lane rightLane = fitLane(rightSI, height);

    return {leftLane, rightLane};
}

Lane LaneDetector::fitLane(
        const std::vector<std::pair<double, double>>& si,
        int height) const
{
    Lane lane;
    if (si.empty()) return lane; // not enough data this frame

    // Arithmetic mean of all slope/intercept pairs collected for this side.
    // A weighted mean (by segment length) would be more robust for production
    // use, but the simple mean works well when combined with ROI filtering.
    const double n        = static_cast<double>(si.size());
    double sumSlope = 0.0, sumIntercept = 0.0;
    for (const auto& [s, b] : si) {
        sumSlope     += s;
        sumIntercept += b;
    }
    const double avgS = sumSlope     / n;
    const double avgB = sumIntercept / n;

    // Extrapolate: given y = avgS·x + avgB  ⟹  x = (y − avgB) / avgS
    const int yBottom = height;
    const int yTop    = static_cast<int>(height * config_.roiHeightRatio);

    lane.bottom = cv::Point(static_cast<int>((yBottom - avgB) / avgS), yBottom);
    lane.top    = cv::Point(static_cast<int>((yTop    - avgB) / avgS), yTop);
    lane.valid  = true;
    return lane;
}

// Stage 6: overlayLanes — blend result onto original frame

cv::Mat LaneDetector::overlayLanes(const cv::Mat& frame,
                                    const Lane& left,
                                    const Lane& right) const
{
    // Draw everything on a black canvas first, then blend so the original
    // frame remains clearly visible underneath.
    cv::Mat overlay = cv::Mat::zeros(frame.size(), frame.type());

    constexpr int kLineThickness = 8;

    // --- Filled polygon between the two lanes (semi-transparent green) ---
    if (left.valid && right.valid) {
        std::vector<cv::Point> poly = {
            left.bottom, left.top,
            right.top,   right.bottom
        };
        cv::fillPoly(overlay,
                     std::vector<std::vector<cv::Point>>{poly},
                     cv::Scalar(0, 80, 0)); // dark green fill
    }

    // --- Lane lines ---
    if (left.valid) {
        // Blue (BGR) for left lane — convention: driver's left
        cv::line(overlay, left.bottom, left.top,
                 cv::Scalar(255, 0, 0), kLineThickness, cv::LINE_AA);
    }
    if (right.valid) {
        // Red (BGR) for right lane — convention: driver's right
        cv::line(overlay, right.bottom, right.top,
                 cv::Scalar(0, 0, 255), kLineThickness, cv::LINE_AA);
    }

    // Blend overlay with original: result = α·frame + (1−α)·overlay
    cv::Mat result;
    cv::addWeighted(frame, config_.blendAlpha,
                    overlay, 1.0 - config_.blendAlpha,
                    0.0, result);

    // Re-draw the lane lines on top of the blend at full opacity so they
    // remain crisp regardless of the alpha value.
    if (left.valid) {
        cv::line(result, left.bottom, left.top,
                 cv::Scalar(255, 50, 50), kLineThickness, cv::LINE_AA);
    }
    if (right.valid) {
        cv::line(result, right.bottom, right.top,
                 cv::Scalar(50, 50, 255), kLineThickness, cv::LINE_AA);
    }

    return result;
}
