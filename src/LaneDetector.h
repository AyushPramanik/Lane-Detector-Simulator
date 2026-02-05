#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <string>

struct LaneConfig {
    // Gaussian blur kernel size (must be odd)
    int gaussianKernelSize{5};

    // Canny edge detection thresholds
    int cannyLow{50};
    int cannyHigh{150};

    // Probabilistic Hough transform parameters
    int houghThreshold{50};      // minimum number of votes (intersections in Hough space)
    int houghMinLineLength{100}; // minimum pixel length of a line segment
    int houghMaxLineGap{50};     // maximum gap (px) between collinear points to bridge

    // Region-of-interest shape (trapezoid, as fraction of frame dimensions)
    // The ROI is centred horizontally and spans from the frame bottom upward.
    double roiBottomWidthRatio{0.90}; // fraction of frame width at bottom edge of ROI
    double roiTopWidthRatio{0.10};    // fraction of frame width at top edge of ROI
    double roiHeightRatio{0.60};      // where the ROI top sits (0 = top, 1 = bottom of frame)

    // Slope filtering — discard near-horizontal lines
    double minSlopeThreshold{0.5};

    // Alpha for the final addWeighted blend (higher = more original frame)
    double blendAlpha{0.75};
};

// Lane — a single extrapolated lane line
struct Lane {
    cv::Point bottom;  // point at the bottom of the frame
    cv::Point top;     // point at the ROI upper boundary
    bool valid{false}; // true when enough line segments were found
};

class LaneDetector {
public:
    explicit LaneDetector(const LaneConfig& config = LaneConfig{});

    // Process a single BGR frame; returns a new frame with lanes overlaid.
    cv::Mat processFrame(const cv::Mat& frame);

    // Runtime config access
    void setConfig(const LaneConfig& config);
    const LaneConfig& getConfig() const noexcept;

private:
    LaneConfig config_;

    \

    // Grayscale conversion + Gaussian blur
    cv::Mat preprocessFrame(const cv::Mat& frame) const;

    // Canny edge map from blurred grayscale
    cv::Mat detectEdges(const cv::Mat& blurred) const;

    // Build a trapezoidal binary mask and AND it with the edge image
    cv::Mat applyROI(const cv::Mat& edges,
                     const std::vector<cv::Point>& roi) const;

    // Run HoughLinesP on the masked edge image
    std::vector<cv::Vec4i> detectLines(const cv::Mat& maskedEdges) const;

    // Classify raw segments into left/right, average each side, extrapolate
    std::pair<Lane, Lane> computeLanes(const std::vector<cv::Vec4i>& lines,
                                       int width, int height) const;

    // Blend the lane overlay onto the original frame
    cv::Mat overlayLanes(const cv::Mat& frame,
                         const Lane& left, const Lane& right) const;

    std::vector<cv::Point> computeROIVertices(int width, int height) const;

    // Fit a single Lane from a collection of (slope, intercept) pairs
    Lane fitLane(const std::vector<std::pair<double, double>>& si,
                 int height) const;
};
