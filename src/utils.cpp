#include "utils.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>

void printUsage(const char* prog) {
    std::cout
        << "\nUsage:\n"
        << "  " << prog << " --video <path>   [options]\n"
        << "  " << prog << " --webcam          [options]\n"
        << "\nInput (one required):\n"
        << "  --video  <file>       Path to input video file (.mp4, .avi, …)\n"
        << "  --webcam              Use the default webcam (device 0)\n"
        << "\nCanny thresholds:\n"
        << "  --canny-low  <int>    Lower hysteresis threshold  (default: 50)\n"
        << "  --canny-high <int>    Upper hysteresis threshold  (default: 150)\n"
        << "\nHough transform:\n"
        << "  --hough-threshold <int>   Minimum votes             (default: 50)\n"
        << "  --hough-min-len   <int>   Minimum segment length px (default: 100)\n"
        << "  --hough-max-gap   <int>   Maximum collinear gap px  (default: 50)\n"
        << "\nROI (fractions of frame size, 0–1):\n"
        << "  --roi-bottom <float>  Bottom width ratio  (default: 0.90)\n"
        << "  --roi-top    <float>  Top width ratio     (default: 0.10)\n"
        << "  --roi-height <float>  ROI vertical start  (default: 0.60)\n"
        << "\nOutput:\n"
        << "  --save [path]         Save processed video (default name: output.avi)\n"
        << "  --debug               Show intermediate windows (edges, ROI)\n"
        << "\nMiscellaneous:\n"
        << "  --help / -h           Print this help and exit\n\n";
}

AppConfig parseArguments(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        throw std::invalid_argument("No arguments provided. Use --video or --webcam.");
    }

    AppConfig cfg;
    bool inputSpecified = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        auto nextStr = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Expected value after '" + arg + "'");
            }
            return std::string(argv[++i]);
        };
        auto nextInt = [&]() -> int {
            return std::stoi(nextStr());
        };
        auto nextDouble = [&]() -> double {
            return std::stod(nextStr());
        };

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--video") {
            cfg.videoPath    = nextStr();
            cfg.useWebcam    = false;
            inputSpecified   = true;
        } else if (arg == "--webcam") {
            cfg.useWebcam    = true;
            inputSpecified   = true;
        } else if (arg == "--canny-low") {
            cfg.laneConfig.cannyLow = nextInt();
        } else if (arg == "--canny-high") {
            cfg.laneConfig.cannyHigh = nextInt();
        } else if (arg == "--hough-threshold") {
            cfg.laneConfig.houghThreshold = nextInt();
        } else if (arg == "--hough-min-len") {
            cfg.laneConfig.houghMinLineLength = nextInt();
        } else if (arg == "--hough-max-gap") {
            cfg.laneConfig.houghMaxLineGap = nextInt();
        } else if (arg == "--roi-bottom") {
            cfg.laneConfig.roiBottomWidthRatio = nextDouble();
        } else if (arg == "--roi-top") {
            cfg.laneConfig.roiTopWidthRatio = nextDouble();
        } else if (arg == "--roi-height") {
            cfg.laneConfig.roiHeightRatio = nextDouble();
        } else if (arg == "--save") {
            cfg.saveOutput = true;
            // Optional path after --save
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.outputPath = std::string(argv[++i]);
            }
        } else if (arg == "--debug") {
            cfg.showDebug = true;
        } else {
            throw std::invalid_argument("Unknown argument: '" + arg + "'");
        }
    }

    if (!inputSpecified) {
        printUsage(argv[0]);
        throw std::invalid_argument("No input specified. Use --video <path> or --webcam.");
    }

    return cfg;
}

FPSCounter::FPSCounter()
    : lastTick_(cv::getTickCount())
    , tickFreq_(cv::getTickFrequency())
{}

void FPSCounter::tick() {
    ++framesSinceUpdate_;
    if (framesSinceUpdate_ >= kUpdateInterval) {
        const int64_t now = cv::getTickCount();
        fps_ = static_cast<double>(kUpdateInterval) * tickFreq_
               / static_cast<double>(now - lastTick_);
        lastTick_          = now;
        framesSinceUpdate_ = 0;
    }
}

double FPSCounter::getFPS() const noexcept { return fps_; }

void FPSCounter::draw(cv::Mat& frame) const {
    const std::string text = "FPS: " + std::to_string(static_cast<int>(fps_ + 0.5));

    // Shadow for readability on any background
    cv::putText(frame, text, cv::Point(12, 34),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(frame, text, cv::Point(10, 32),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 230, 0), 2, cv::LINE_AA);
}
