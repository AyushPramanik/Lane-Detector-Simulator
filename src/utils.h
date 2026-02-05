#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "LaneDetector.h"


struct AppConfig {
    bool        useWebcam{false};
    std::string videoPath;

    // Output recording
    bool        saveOutput{false};
    std::string outputPath{"output.avi"};

    // Show intermediate debug windows
    bool showDebug{false};

    LaneConfig laneConfig;
};

AppConfig parseArguments(int argc, char* argv[]);

void printUsage(const char* programName);

class FPSCounter {
public:
    FPSCounter();

    // Call once per processed frame
    void tick();

    double getFPS() const noexcept;

    // Render FPS
    void draw(cv::Mat& frame) const;

private:
    int64_t lastTick_;
    double  tickFreq_;
    double  fps_{0.0};
    int     framesSinceUpdate_{0};

    // Recalculate every N frames
    static constexpr int kUpdateInterval{15};
};
