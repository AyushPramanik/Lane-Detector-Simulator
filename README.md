# Lane Detector Simulator

A production-quality, real-time lane detection simulator built in **C++** using only
classical computer vision techniques from **OpenCV** — no machine learning, no Python.

---

## Pipeline Overview

```
Frame
  │
  ▼
┌─────────────────┐
│  Grayscale +    │  Remove colour channels; Gaussian blur suppresses noise
│  Gaussian Blur  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Canny Edge     │  Gradient magnitude → non-max suppression → hysteresis
│  Detection      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ROI Mask       │  Trapezoidal polygon discards sky/scenery; bitwise-AND
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hough Lines P  │  Probabilistic Hough finds line segments in edge space
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Lane Averaging │  Classify left/right by slope sign; average slope &
│  & Extrap.      │  intercept; extrapolate to full lane extent
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Overlay Blend  │  addWeighted() merges semi-transparent fill + crisp lines
└─────────────────┘
```

---

## Project Structure

```
lane-detector-simulator/
├── CMakeLists.txt          # CMake build definition
├── README.md               # This file
└── src/
    ├── main.cpp            # Entry point: CLI parsing, capture loop
    ├── LaneDetector.h      # Pipeline class declaration + LaneConfig struct
    ├── LaneDetector.cpp    # Full pipeline implementation
    ├── utils.h             # AppConfig, FPSCounter declarations
    └── utils.cpp           # Argument parser, FPS counter
```

---

## Prerequisites

| Dependency | Minimum version | Notes |
|------------|-----------------|-------|
| C++ compiler | C++17 | GCC 9+, Clang 10+, MSVC 2019+ |
| CMake | 3.16 | |
| OpenCV | 4.x | Modules: core, imgproc, highgui, videoio |

### Install OpenCV

**macOS (Homebrew)**
```bash
brew install opencv
```

**Ubuntu / Debian**
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

**Windows (vcpkg)**
```powershell
vcpkg install opencv4[core,highgui,imgproc,videoio]:x64-windows
```

**From source** (any platform)
```bash
git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_LIST=core,imgproc,highgui,videoio ..
make -j$(nproc)
sudo make install
```

---

## Building

```bash
git clone <repo-url>
cd lane-detector-simulator

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

If OpenCV is installed to a non-standard prefix, pass its CMake directory:
```bash
cmake -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4 ..
```

The compiled binary is `build/lane_detector`.

---

## Running

### Video file input
```bash
./lane_detector --video road.mp4
```

### Sample videos
```bash
./lane_detector --video videos/solidWhiteRight.mp4
./lane_detector --video videos/solidYellowLeft.mp4
```

### Webcam
```bash
./lane_detector --webcam
```

### Save output video
```bash
./lane_detector --video road.mp4 --save output.avi
```

### Tune Canny thresholds
```bash
./lane_detector --video road.mp4 --canny-low 40 --canny-high 120
```

### Show intermediate debug windows
```bash
./lane_detector --video road.mp4 --debug
```

### Full example with all options
```bash
./lane_detector \
  --video road.mp4 \
  --canny-low 40 --canny-high 130 \
  --hough-threshold 40 \
  --hough-min-len 80 \
  --hough-max-gap 60 \
  --roi-height 0.58 \
  --save lanes_output.avi \
  --debug
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--video <path>` | — | Input video file |
| `--webcam` | — | Use default webcam |
| `--canny-low <int>` | 50 | Canny lower threshold |
| `--canny-high <int>` | 150 | Canny upper threshold |
| `--hough-threshold <int>` | 50 | Min Hough votes |
| `--hough-min-len <int>` | 100 | Min segment length (px) |
| `--hough-max-gap <int>` | 50 | Max collinear gap (px) |
| `--roi-bottom <float>` | 0.90 | ROI bottom width (fraction) |
| `--roi-top <float>` | 0.10 | ROI top width (fraction) |
| `--roi-height <float>` | 0.60 | ROI vertical start (fraction) |
| `--save [path]` | output.avi | Save processed video |
| `--debug` | off | Show edge/ROI windows |
| `--help` | — | Print usage |

**Press `q` or `ESC` to quit.**

---

## Visual Output

| Element | Colour | Meaning |
|---------|--------|---------|
| Left lane line | Blue | Driver's left lane marking |
| Right lane line | Red | Driver's right lane marking |
| Lane fill | Semi-transparent dark green | Drivable lane area |
| FPS counter | Green (top-left) | Processing frame rate |

---

## Pipeline Explained

### 1. Preprocessing
Converting to grayscale drops two of three colour channels, halving memory
bandwidth for all downstream operations.  A 5×5 Gaussian kernel then blurs
high-frequency pixel noise that would otherwise produce many spurious edges.
Larger kernels remove more noise but also soften real lane-marking boundaries.

### 2. Canny Edge Detection
Canny applies a Sobel filter in both axes to estimate the gradient magnitude at
every pixel.  Non-maximum suppression thins edges to one pixel wide.
Hysteresis thresholding then accepts a pixel as an edge only if it exceeds the
high threshold *or* is connected (directly or transitively) to a pixel that
does.  This avoids fragmenting real edges while suppressing isolated noise.
Typical road footage works well with `cannyLow=50, cannyHigh=150`; increase
both values if too many spurious edges survive.

### 3. ROI Masking
A trapezoidal polygon is drawn on a black mask and AND-ed with the edge image.
The trapezoid is narrow at the top (vanishing-point region) and wide at the
bottom (near the vehicle).  This discards irrelevant edges from the sky,
trees, and vehicle hood, and focuses Hough processing on the actual road.
**Why it matters:** Without ROI masking, Hough will find dozens of false-positive
lines from guardrails, signage, and horizon features, overwhelming the true
lane lines.

### 4. Probabilistic Hough Transform
HoughLinesP maps each edge pixel into Hough parameter space (ρ, θ) and looks
for cells with many votes (line intersections).  The probabilistic variant
randomly samples edge pixels, which is faster than the classical transform and
directly returns line segments rather than infinite lines.  The `threshold`
parameter controls how many edge pixels must align before the transform reports
a line; `minLineLength` discards short disconnected blobs; `maxLineGap` bridges
gaps in dashed lane markings.

### 5. Lane Averaging & Extrapolation
Each detected segment has a slope `m` and y-intercept `b` (line equation
`y = mx + b`).  In image coordinates y increases downward, so the left lane
has a negative slope and the right lane has a positive slope.  Near-horizontal
segments (`|m| < 0.5`) are discarded as noise.  All remaining segments for each
side are averaged to produce a single representative line, then extrapolated
from the bottom of the frame to the top of the ROI using `x = (y − b) / m`.

---

## Known Limitations & Trade-offs

| Limitation | Root cause | Mitigation |
|------------|-----------|------------|
| Fails on sharp curves | Linear model (straight line fit) | Polynomial or spline fitting |
| Sensitive to shadows & glare | Edge-based approach reacts to intensity changes | HSV colour masking before edge detection |
| Lane loss in low contrast | Canny thresholds fixed | Adaptive thresholding or CLAHE pre-processing |
| Slope averaging is unweighted | Simple mean | Weight by segment length or confidence score |
| ROI is hard-coded as fraction | Camera mount varies | Calibrated homography-based ROI |

---

## Future Improvements

- **Curved lane detection** — fit a quadratic/cubic polynomial with `np::polyfit`-equivalent logic or use a sliding-window histogram approach.
- **Temporal smoothing** — use an EMA (exponential moving average) on slope/intercept across frames to eliminate flicker.
- **Colour-space masking** — pre-filter white and yellow pixels in HLS space before grayscale conversion to boost robustness.
- **Calibration & perspective transform** — warp the frame to a bird's-eye view for more accurate polynomial fitting and width estimation.
- **Deep learning upgrade** — replace the classical pipeline with a lightweight segmentation network (e.g., LaneNet, SCNN) for handling complex scenarios.
- **Lane departure warning** — compute lane centre offset and trigger alerts when the vehicle drifts.

---

## License

MIT License.  See `LICENSE` for details.
