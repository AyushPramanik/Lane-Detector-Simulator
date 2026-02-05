#include <iostream>
#include <stdexcept>
#include <string>
#include <opencv2/opencv.hpp>
#include "LaneDetector.h"
#include "utils.h"

static cv::VideoCapture openSource(const AppConfig& cfg) {
    cv::VideoCapture cap;

    if (cfg.useWebcam) {
        cap.open(0); // device index 0 = default webcam
        if (!cap.isOpened()) {
            throw std::runtime_error("Could not open webcam (device 0).");
        }
        std::cout << "[info] Input: webcam\n";
    } else {
        cap.open(cfg.videoPath);
        if (!cap.isOpened()) {
            throw std::runtime_error("Could not open video file: " + cfg.videoPath);
        }
        std::cout << "[info] Input: " << cfg.videoPath << "\n";
    }

    return cap;
}

static cv::VideoWriter openWriter(const AppConfig& cfg, const cv::VideoCapture& cap) {
    cv::VideoWriter writer;

    if (!cfg.saveOutput) return writer;

    const int    fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    const double fps    = cap.get(cv::CAP_PROP_FPS);
    const int    width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    writer.open(cfg.outputPath, fourcc,
                fps > 0 ? fps : 25.0,       // Fallback fps for webcam
                cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "[warn] Could not open VideoWriter for: " << cfg.outputPath
                  << ". Output will not be saved.\n";
    } else {
        std::cout << "[info] Saving output to: " << cfg.outputPath << "\n";
    }

    return writer;
}

static void showDebugWindows(const cv::Mat& frame, const LaneDetector& detector) {
    const int    h   = frame.rows;
    const int    w   = frame.cols;
    const auto&  cfg = detector.getConfig();

    cv::Mat gray, blurred, edges;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    const int k = cfg.gaussianKernelSize | 1;
    cv::GaussianBlur(gray, blurred, cv::Size(k, k), 0);
    cv::Canny(blurred, edges, cfg.cannyLow, cfg.cannyHigh);

    // Draw ROI outline on a copy of the original
    cv::Mat roiViz = frame.clone();
    const int cx         = w / 2;
    const int bottomY    = h;
    const int topY       = static_cast<int>(h * cfg.roiHeightRatio);
    const int bottomHalf = static_cast<int>(w * cfg.roiBottomWidthRatio / 2.0);
    const int topHalf    = static_cast<int>(w * cfg.roiTopWidthRatio    / 2.0);
    std::vector<cv::Point> roiPts = {
        {cx - bottomHalf, bottomY}, {cx - topHalf, topY},
        {cx + topHalf,    topY},    {cx + bottomHalf, bottomY}
    };
    const cv::Point* ptr  = roiPts.data();
    const int        npts = static_cast<int>(roiPts.size());
    cv::polylines(roiViz, &ptr, &npts, 1, /*closed=*/true,
                  cv::Scalar(0, 255, 255), 2);

    cv::imshow("Debug: Edges",       edges);
    cv::imshow("Debug: ROI outline", roiViz);
}

int main(int argc, char* argv[]) {
    // Parse arguments
    AppConfig cfg;
    try {
        cfg = parseArguments(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }

    // Open video source
    cv::VideoCapture cap;
    try {
        cap = openSource(cfg);
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }

    // Optional output writer
    cv::VideoWriter writer = openWriter(cfg, cap);

    // Print active parameters
    const LaneConfig& lc = cfg.laneConfig;
    std::cout << "[info] Canny thresholds    : " << lc.cannyLow  << " / " << lc.cannyHigh << "\n"
              << "[info] Hough threshold     : " << lc.houghThreshold    << "\n"
              << "[info] Hough min length    : " << lc.houghMinLineLength << "\n"
              << "[info] Hough max gap       : " << lc.houghMaxLineGap    << "\n"
              << "[info] ROI height ratio    : " << lc.roiHeightRatio     << "\n"
              << "[info] Press 'q' or ESC to quit.\n\n";

    // Build detector
    LaneDetector detector(cfg.laneConfig);
    FPSCounter   fpsCounter;
    cv::Mat      frame;

    // Main processing loop
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "[info] End of stream.\n";
            break;
        }

        // Run the full lane-detection pipeline
        cv::Mat result = detector.processFrame(frame);

        // Overlay FPS counter
        fpsCounter.tick();
        fpsCounter.draw(result);

        // Display
        cv::imshow("Lane Detector", result);

        // Optional debug windows
        if (cfg.showDebug) {
            showDebugWindows(frame, detector);
        }

        // Save
        if (writer.isOpened()) {
            writer.write(result);
        }

        // Wait 1 ms; quit on 'q' (ASCII 113) or ESC (ASCII 27)
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            std::cout << "[info] Quit requested by user.\n";
            break;
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    cv::destroyAllWindows();

    return 0;
}
