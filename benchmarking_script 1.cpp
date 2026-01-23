#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <rpp.h>
#include <rppdefs.h>
#include <dirent.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>
#include <thread>
#include <cstdlib>

// Test image paths
#define GRAY_IMAGE_PATH "1080p_128images_dataset/"
#define RGB_IMAGE_PATH "1080p_128images_dataset/"

#define NUM_THREADS 128

using namespace std;
using namespace cv;
using namespace chrono;

// Loads all valid images from a directory into a vector.
vector<Mat> loadBatchImages(const string& directory, int& batchSize, int& maxWidth, int& maxHeight, bool isColor) {
    vector<Mat> images;
    DIR* dir;
    struct dirent* entry;

    // Initialize max dimensions
    maxWidth = 0;
    maxHeight = 0;

    // Try opening the directory
    if ((dir = opendir(directory.c_str())) == NULL) {
        cerr << "Could not open directory: " << directory << endl;
        return images;
    }

    // Read all entries in the directory
    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;

        // Skip "." and ".."
        if (filename == "." || filename == "..") continue;

        // Check file extension for common image formats
        string ext = filename.substr(filename.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != "jpg" && ext != "jpeg" && ext != "png" && ext != "bmp" && ext != "tiff")
            continue;

        // Build full file path and load image
        string filePath = directory + "/" + filename;
        Mat img = imread(filePath, isColor ? IMREAD_COLOR : IMREAD_GRAYSCALE);

        // If image loaded successfully
        if (!img.empty()) {
            maxWidth = max(maxWidth, img.cols);   // Update max width
            maxHeight = max(maxHeight, img.rows); // Update max height
            images.push_back(std::move(img));     // Add to list (use move semantics)
        } else {
            cerr << "Warning: Could not read image " << filePath << endl;
        }
    }

    closedir(dir);                  // Close directory stream
    batchSize = images.size();      // Set output batch size
    return images;                  // Return the list of loaded images
}

// Helper function to initialize descriptors and ROI
void initializeDescriptorsAndRoi(const vector<Mat>& imgs, int maxWidth, int maxHeight,
    vector<RpptDesc>& srcDescs, vector<RpptDesc>& dstDescs,
    vector<RpptROI>& rois)
{
    int channels = imgs[0].channels();
    int batchSize = imgs.size();

    for (int i = 0; i < batchSize; ++i)
    {
        const Mat& img = imgs[i];

        // Set ROI
        rois[i].xywhROI.xy.x = 0;
        rois[i].xywhROI.xy.y = 0;
        rois[i].xywhROI.roiWidth = img.cols;
        rois[i].xywhROI.roiHeight = img.rows;

        // Set descriptor dimensions
        srcDescs[i].h = dstDescs[i].h = img.rows;
        srcDescs[i].w = dstDescs[i].w = img.cols;
        srcDescs[i].c = dstDescs[i].c = channels;
        srcDescs[i].n = dstDescs[i].n = 1;
        srcDescs[i].dataType = dstDescs[i].dataType = RpptDataType::U8;
        srcDescs[i].strides.nStride = dstDescs[i].strides.nStride = img.rows * img.cols * channels;

        if (channels == 3)
        {
            // NHWC layout
            srcDescs[i].strides.hStride = dstDescs[i].strides.hStride = img.cols * channels;
            srcDescs[i].strides.wStride = dstDescs[i].strides.wStride = channels;
            srcDescs[i].strides.cStride = dstDescs[i].strides.cStride = 1;
            srcDescs[i].layout = dstDescs[i].layout = RpptLayout::NHWC;
        }
        else
        {
            // NCHW layout
            srcDescs[i].strides.hStride = dstDescs[i].strides.hStride = img.cols;
            srcDescs[i].strides.wStride = dstDescs[i].strides.wStride = 1;
            srcDescs[i].strides.cStride = dstDescs[i].strides.cStride = img.cols * img.rows;
            srcDescs[i].layout = dstDescs[i].layout = RpptLayout::NCHW;
        }
    }
}

void benchmarkRPP_gaussian_filter(const vector<Mat>& imgs, int maxWidth, int maxHeight, int kernelSize)
{
    int batchSize = imgs.size();
    if (batchSize == 0) {
        cerr << "Error: No images to process!" << endl;
        return;
    }

    int channels = imgs[0].channels();

    // Allocate output images
    vector<Mat> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
    output[i] = Mat(imgs[i].rows, imgs[i].cols, imgs[i].type());

    // Initialize descriptors and ROIs
    vector<RpptDesc> srcDescs(batchSize), dstDescs(batchSize);
    vector<RpptROI> rois(batchSize);
    initializeDescriptorsAndRoi(imgs, maxWidth, maxHeight, srcDescs, dstDescs, rois);

    rppHandle_t handle;
    rppCreate(&handle, 1);
    vector<float> stdDevTensor(batchSize, 5.0f);

    auto start = high_resolution_clock::now();

    // Run benchmark
    for (int k = 0; k < 100; ++k) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < batchSize; ++i) {
            rppt_gaussian_filter_host(imgs[i].data, &srcDescs[i], output[i].data, &dstDescs[i], &stdDevTensor[i],
                                      kernelSize, &rois[i], RpptRoiType::XYWH, handle);
        }
    }

    auto end = high_resolution_clock::now();

    cout << "RPP Gauusian Filter (Avg per run, " << batchSize << " images, "
    << (channels == 3 ? "RGB" : "Grayscale") << "): "
    << duration_cast<milliseconds>(end - start).count() / 100.0 << " ms" << endl;

    rppDestroy(handle);
}

void benchmarkRPP_median_filter(const vector<Mat>& imgs, int maxWidth, int maxHeight, int kernelSize)
{
    int batchSize = imgs.size();
    if (batchSize == 0) {
        cerr << "Error: No images to process!" << endl;
        return;
    }

    int channels = imgs[0].channels();

    // Allocate output images
    vector<Mat> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
        output[i] = Mat(imgs[i].rows, imgs[i].cols, imgs[i].type());

    // Initialize descriptors and ROIs
    vector<RpptDesc> srcDescs(batchSize), dstDescs(batchSize);
    vector<RpptROI> rois(batchSize);
    initializeDescriptorsAndRoi(imgs, maxWidth, maxHeight, srcDescs, dstDescs, rois);

    rppHandle_t handle;
    rppCreate(&handle, 1);

    auto start = high_resolution_clock::now();

    // Run benchmark
    for (int k = 0; k < 100; ++k) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < batchSize; ++i) {
            rppt_median_filter_host(imgs[i].data, &srcDescs[i], output[i].data, &dstDescs[i], 
                                    kernelSize, RpptImageBorderType::REPLICATE, &rois[i], 
                                    RpptRoiType::XYWH, handle);
        }
    }

    auto end = high_resolution_clock::now();

    cout << "RPP Median Filter (Avg per run, " << batchSize << " images, "
    << (channels == 3 ? "RGB" : "Grayscale") << ", kernel size " << kernelSize << "): "
    << duration_cast<milliseconds>(end - start).count() / 100.0 << " ms" << endl;

    rppDestroy(handle);
}

void benchmarkRPP_box_filter(const vector<Mat>& imgs, int maxWidth, int maxHeight, int kernelSize)
{
    int batchSize = imgs.size();
    if (batchSize == 0) {
        cerr << "Error: No images to process!" << endl;
        return;
    }

    int channels = imgs[0].channels();

    // Allocate output images
    vector<Mat> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
        output[i] = Mat(imgs[i].rows, imgs[i].cols, imgs[i].type());

    // Initialize descriptors and ROIs
    vector<RpptDesc> srcDescs(batchSize), dstDescs(batchSize);
    vector<RpptROI> rois(batchSize);
    initializeDescriptorsAndRoi(imgs, maxWidth, maxHeight, srcDescs, dstDescs, rois);

    rppHandle_t handle;
    rppCreate(&handle, 1);

    auto start = high_resolution_clock::now();

    // Run benchmark
    for (int k = 0; k < 100; ++k) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < batchSize; ++i) {
            rppt_box_filter_host(imgs[i].data, &srcDescs[i], output[i].data, &dstDescs[i], 
                                 kernelSize, RpptImageBorderType::REPLICATE, &rois[i], 
                                 RpptRoiType::XYWH, handle);
        }
    }

    auto end = high_resolution_clock::now();

    cout << "RPP Box Filter (Avg per run, " << batchSize << " images, "
    << (channels == 3 ? "RGB" : "Grayscale") << ", kernel size " << kernelSize << "): "
    << duration_cast<milliseconds>(end - start).count() / 100.0 << " ms" << endl;

    rppDestroy(handle);
}

void benchmarkRPP_brightness(const vector<Mat>& imgs, int maxWidth, int maxHeight, float alpha, float beta)
{
    int batchSize = imgs.size();
    if (batchSize == 0) {
        cerr << "Error: No images to process!" << endl;
        return;
    }

    int channels = imgs[0].channels();

    // Allocate output images
    vector<Mat> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
        output[i] = Mat(imgs[i].rows, imgs[i].cols, imgs[i].type());

    // Initialize descriptors and ROIs
    vector<RpptDesc> srcDescs(batchSize), dstDescs(batchSize);
    vector<RpptROI> rois(batchSize);
    initializeDescriptorsAndRoi(imgs, maxWidth, maxHeight, srcDescs, dstDescs, rois);

    // Brightness parameters
    vector<float> alphaValues(batchSize, alpha);
    vector<float> betaValues(batchSize, beta);

    rppHandle_t handle;
    rppCreate(&handle, 1);

    auto start = high_resolution_clock::now();

    // Run benchmark
    for (int k = 0; k < 100; ++k) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < batchSize; ++i) {
            rppt_brightness_host(imgs[i].data, &srcDescs[i], output[i].data, &dstDescs[i], 
                                 alphaValues[i], betaValues[i], &rois[i], 
                                 RpptRoiType::XYWH, handle);
        }
    }

    auto end = high_resolution_clock::now();

    cout << "RPP Brightness (Avg per run, " << batchSize << " images, "
    << (channels == 3 ? "RGB" : "Grayscale") << "): "
    << duration_cast<milliseconds>(end - start).count() / 100.0 << " ms" << endl;

    rppDestroy(handle);
}

void benchmarkOpenCV_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize)
{
    vector<Mat> output(imgs.size());
    double sigma = 5.0; // Standard deviation to match RPP's stdDevTensor
    
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t j = 0; j < imgs.size(); j++) {
            GaussianBlur(imgs[j], output[j], Size(kernelSize, kernelSize), sigma, sigma, cv::BorderTypes::BORDER_REPLICATE);
        }
    }

    auto end = high_resolution_clock::now();
    cout << "OpenCV GaussianFilter (Avg per run, " << imgs.size() << " images, " << (isColor ? "RGB" : "Grayscale") << "): "
         << duration_cast<milliseconds>(end - start).count() / 100.0 << " ms" << endl;
}

void benchmarkOpenCV_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize)
{
    vector<Mat> output(imgs.size());
    
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t j = 0; j < imgs.size(); j++) {
            medianBlur(imgs[j], output[j], kernelSize);
        }
    }

    auto end = high_resolution_clock::now();
    cout << "OpenCV MedianFilter (Avg per run, " << imgs.size() << " images, " 
         << (isColor ? "RGB" : "Grayscale") << ", kernel size " << kernelSize << "): "
         << duration_cast<milliseconds>(end - start).count() / 100.0 << " ms" << endl;
}

void benchmarkOpenCV_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize)
{
    vector<Mat> output(imgs.size());
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t j = 0; j < imgs.size(); j++) {
            boxFilter(imgs[j], output[j], -1, Size(kernelSize, kernelSize), cv::Point(-1, -1), true, cv::BorderTypes::BORDER_REPLICATE);
        }
    }

    auto end = high_resolution_clock::now();
    cout << "OpenCV BoxFilter (Avg per run, " << imgs.size() << " images, " << (isColor ? "RGB" : "Grayscale") << ", kernel size " << kernelSize << "): "
         << duration_cast<milliseconds>(end - start).count() / 100.0 << " ms" << endl;
}

void benchmarkOpenCV_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta) {
    vector<Mat> output(imgs.size());
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t j = 0; j < imgs.size(); j++) {
            imgs[j].convertTo(output[j], -1, alpha, beta);
        }
    }

    auto end = high_resolution_clock::now();
    cout << "OpenCV Brightness (Avg per run, " << imgs.size() << " images, " << (isColor ? "RGB" : "Grayscale") << "): "
         << duration_cast<milliseconds>(end - start).count() / 100.0 << " ms" << endl;
}

int main() {
    // Initialization of batch sizes and maximum dimensions
    int batchSizeGray = 0, maxWidthGray = 0, maxHeightGray = 0;
    int batchSizeRGB = 0, maxWidthRGB = 0, maxHeightRGB = 0;

    // Load grayscale images
    vector<Mat> imgsGray = loadBatchImages(GRAY_IMAGE_PATH, batchSizeGray, maxWidthGray, maxHeightGray, false);
    
    // Load RGB images
    vector<Mat> imgsRGB = loadBatchImages(RGB_IMAGE_PATH, batchSizeRGB, maxWidthRGB, maxHeightRGB, true);

    // Error check if no images are loaded
    if (imgsGray.empty() && imgsRGB.empty()) {
        cerr << "No images found in the directories!" << endl;
        return -1;
    }

    // Brightness Params
    float alpha = 1.2f, beta = 20.0f;
    //contrast Params
    float contrastFactor = 2.96f, contrastCenter = 128.0f;
    // Filter Params
    float kernelSize = 3;
    int medianKernelSize = 3;  // Median filter kernel size (must be odd)

    // Benchmark for grayscale images
    if (!imgsGray.empty()) {
        std::cout<<"\n Running Gaussian Filter Grayscale \n";
        // benchmarkOpenCV_GaussianFilter(imgsGray, false, kernelSize);
        // benchmarkRPP_gaussian_filter(imgsGray, maxWidthGray, maxHeightGray, kernelSize);
        
        // std::cout<<"\n Running Median Filter Grayscale \n";
        // benchmarkOpenCV_MedianFilter(imgsGray, false, medianKernelSize);
        // benchmarkRPP_median_filter(imgsGray, maxWidthGray, maxHeightGray, medianKernelSize);

        std::cout<<"\n Running Box Filter Grayscale \n";
        benchmarkOpenCV_BoxFilter(imgsGray, false, kernelSize);
        benchmarkRPP_box_filter(imgsGray, maxWidthGray, maxHeightGray, kernelSize);

        std::cout<<"\n Running Brightness Grayscale \n";
        benchmarkOpenCV_Brightness(imgsGray, false, alpha, beta);
        benchmarkRPP_brightness(imgsGray, maxWidthGray, maxHeightGray, alpha, beta);
    }

    if (!imgsRGB.empty()) {
        std::cout<<"\n Running Gaussian Filter RGB \n";
        // benchmarkOpenCV_GaussianFilter(imgsRGB, true, kernelSize);
        // benchmarkRPP_gaussian_filter(imgsRGB, maxWidthRGB, maxHeightRGB, kernelSize);
        
        std::cout<<"\n Running Median Filter RGB \n";
        // benchmarkOpenCV_MedianFilter(imgsRGB, true, medianKernelSize);
        // benchmarkRPP_median_filter(imgsRGB, maxWidthRGB, maxHeightRGB, medianKernelSize);

        std::cout<<"\n Running Box Filter RGB \n";
        benchmarkOpenCV_BoxFilter(imgsRGB, true, kernelSize);
        benchmarkRPP_box_filter(imgsRGB, maxWidthRGB, maxHeightRGB, kernelSize);

        std::cout<<"\n Running Brightness RGB \n";
        benchmarkOpenCV_Brightness(imgsRGB, true, alpha, beta);
        benchmarkRPP_brightness(imgsRGB, maxWidthRGB, maxHeightRGB, alpha, beta);
    }

    return 0;
}
