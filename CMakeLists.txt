cmake_minimum_required(VERSION 3.16)
project(LaneDetectorSimulator
        VERSION 1.0.0
        DESCRIPTION "Classical CV lane detection simulator using OpenCV"
        LANGUAGES CXX)

# ---- Language standard -----------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Default build type ----------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---- Compiler warnings & optimisation flags --------------------------------
if(MSVC)
    add_compile_options(/W4 /WX)
    add_compile_options("$<$<CONFIG:Release>:/O2>")
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # -Werror in CI; drop locally if needed
    # add_compile_options(-Werror)
    add_compile_options("$<$<CONFIG:Release>:-O3>")
    add_compile_options("$<$<CONFIG:Debug>:-g3;-fsanitize=address,undefined>")
    add_link_options("$<$<CONFIG:Debug>:-fsanitize=address,undefined>")
endif()

# ---- Find OpenCV -----------------------------------------------------------
# Hint: set OpenCV_DIR if OpenCV is installed to a non-standard prefix, e.g.:
#   cmake -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 ..
find_package(OpenCV 4 REQUIRED
             COMPONENTS core imgproc highgui videoio)

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "OpenCV include : ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs    : ${OpenCV_LIBS}")

# ---- Source files ----------------------------------------------------------
set(LANE_DETECTOR_SOURCES
    src/main.cpp
    src/LaneDetector.cpp
    src/utils.cpp
)

set(LANE_DETECTOR_HEADERS
    src/LaneDetector.h
    src/utils.h
)

# ---- Executable ------------------------------------------------------------
add_executable(lane_detector
               ${LANE_DETECTOR_SOURCES}
               ${LANE_DETECTOR_HEADERS})

target_include_directories(lane_detector
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(lane_detector
    PRIVATE
        ${OpenCV_LIBS})

# Propagate the project version as a compile definition (optional, useful for
# printing version strings at runtime).
target_compile_definitions(lane_detector
    PRIVATE
        PROJECT_VERSION="${PROJECT_VERSION}")

# ---- Install ---------------------------------------------------------------
install(TARGETS lane_detector
        RUNTIME DESTINATION bin)

# ---- Summary ---------------------------------------------------------------
message(STATUS "")
message(STATUS "  ┌────────────────────────────────────────┐")
message(STATUS "  │  Lane Detector Simulator v${PROJECT_VERSION}       │")
message(STATUS "  │  Compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  │  OpenCV   : ${OpenCV_VERSION}                       │")
message(STATUS "  │  Target   : lane_detector               │")
message(STATUS "  └────────────────────────────────────────┘")
message(STATUS "")
