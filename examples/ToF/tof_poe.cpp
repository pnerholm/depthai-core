#include <iostream>
#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>
#include "depthai-shared/log/LogLevel.hpp"
#include "depthai/device/Device.hpp"

// Function to create ToF configuration
std::shared_ptr<dai::ToFConfig> createConfig(dai::RawToFConfig configRaw) {
    auto config = std::make_shared<dai::ToFConfig>();
    config->set(std::move(configRaw));
    return config;
}

// Function to colorize depth data for visualization
cv::Mat colorizeDepth(const cv::Mat& frameDepth) {
    cv::Mat invalidMask = (frameDepth == 0);
    cv::Mat depthFrameColor;

    try {
        double minDepth, maxDepth;
        cv::minMaxIdx(frameDepth, &minDepth, &maxDepth, nullptr, nullptr, ~invalidMask);

        if (minDepth == maxDepth) {
            depthFrameColor = cv::Mat::zeros(frameDepth.size(), CV_8UC3);
            return depthFrameColor;
        }

        cv::Mat logDepth;
        frameDepth.convertTo(logDepth, CV_32F);
        cv::log(logDepth, logDepth);
        logDepth.setTo(log(minDepth), invalidMask);
        cv::normalize(logDepth, logDepth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(logDepth, depthFrameColor, cv::COLORMAP_JET);
        depthFrameColor.setTo(cv::Scalar(0, 0, 0), invalidMask);
    } catch (const std::exception& e) {
        depthFrameColor = cv::Mat::zeros(frameDepth.size(), CV_8UC3);
    }

    return depthFrameColor;
}

int main() {
    dai::Pipeline pipeline;

    // Set up the camera node
    auto camTof = pipeline.create<dai::node::Camera>();
    camTof->setFps(30);
    camTof->setBoardSocket(dai::CameraBoardSocket::CAM_A);

    // Set up the ToF node
    auto tofNode = pipeline.create<dai::node::ToF>();
    tofNode->setNumShaves(1);

    // Configure the ToF node
    auto tofConfig = tofNode->initialConfig.get();
    tofConfig.enablePhaseShuffleTemporalFilter = true;
    tofConfig.phaseUnwrappingLevel = 4;
    tofConfig.phaseUnwrapErrorThreshold = 300;
    tofNode->initialConfig.set(tofConfig);

    // Link the camera to the ToF node
    camTof->raw.link(tofNode->input);

    // Create an XLinkOut node to send data to the host
    auto xout = pipeline.create<dai::node::XLinkOut>();
    xout->setStreamName("depth");
    tofNode->depth.link(xout->input);

    // Specify the IP address of the POE device
    dai::DeviceInfo deviceInfo("169.254.1.101");

    // Connect to the device
    dai::Device::Config config;
    config.board.logVerbosity = dai::LogLevel::TRACE;
    config.board.logDevicePrints = true;
    dai::Device device(config,deviceInfo);
    device.startPipeline(pipeline);
    //dai::Device device(pipeline, deviceInfo);
    //device.setLogOutputLevel(dai::LogLevel::TRACE);

    std::cout << "Connected cameras: " << device.getConnectedCameraFeatures().size() << std::endl;

    // Get the output queue
    auto qDepth = device.getOutputQueue("depth");

    while (true) {
        auto imgFrame = qDepth->get<dai::ImgFrame>(); // Blocking call
        auto depthColorized = colorizeDepth(imgFrame->getFrame());
        cv::imshow("Colorized Depth", depthColorized);

        // Exit on 'q' or 'Esc' key press
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    }

    return 0;
}

