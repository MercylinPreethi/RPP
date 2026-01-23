#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <dirent.h>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  // SSE4.1
#include <immintrin.h>  // AVX2

#define NUM_ITERATIONS 100

using namespace std;
using namespace cv;
using namespace chrono;

typedef unsigned char Rpp8u;
typedef int Rpp32s;
typedef unsigned int Rpp32u;

enum class RpptLayout { NHWC, NCHW };
enum class RpptRoiType { XYWH, LTRB };
enum RppStatus { RPP_SUCCESS = 0, RPP_ERROR = -1 };

struct RpptImagePatch { Rpp32s x; Rpp32s y; };
struct RpptXYWH { RpptImagePatch xy; Rpp32s roiWidth; Rpp32s roiHeight; };
struct RpptLTRB { RpptImagePatch lt; RpptImagePatch rb; };
struct RpptROI {
    union {
        RpptXYWH xywhROI;
        RpptLTRB ltrbROI;
    };
};
typedef RpptROI* RpptROIPtr;

struct RpptStride { Rpp32u nStride; Rpp32u hStride; Rpp32u wStride; Rpp32u cStride; };
struct RpptDesc { Rpp32u n; Rpp32u c; Rpp32u h; Rpp32u w; RpptLayout layout; RpptStride strides; };
typedef RpptDesc* RpptDescPtr;

struct RppLayoutParams { Rpp32u bufferMultiplier; Rpp32u channelParam; };


inline void rpp_load48_u8pkd3_to_u8pln3(Rpp8u *srcPtr, __m128i *px)
{
    __m128i pxSrc[8];
    __m128i pxMask = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15);
    __m128i pxMaskRGB = _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);

    pxSrc[0] = _mm_loadu_si128((__m128i *)srcPtr);
    pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));
    pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));
    pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));
    pxSrc[0] = _mm_shuffle_epi8(pxSrc[0], pxMask);
    pxSrc[1] = _mm_shuffle_epi8(pxSrc[1], pxMask);
    pxSrc[2] = _mm_shuffle_epi8(pxSrc[2], pxMask);
    pxSrc[3] = _mm_shuffle_epi8(pxSrc[3], pxMask);
    pxSrc[4] = _mm_unpacklo_epi8(pxSrc[0], pxSrc[1]);
    pxSrc[5] = _mm_unpacklo_epi8(pxSrc[2], pxSrc[3]);
    pxSrc[6] = _mm_unpackhi_epi8(pxSrc[0], pxSrc[1]);
    pxSrc[7] = _mm_unpackhi_epi8(pxSrc[2], pxSrc[3]);
    px[0] = _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB);
    px[1] = _mm_shuffle_epi8(_mm_unpackhi_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB);
    px[2] = _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[6], pxSrc[7]), pxMaskRGB);
}

inline void rpp_store48_u8pln3_to_u8pln3(Rpp8u *dstPtrR, Rpp8u *dstPtrG, Rpp8u *dstPtrB, __m128i *px)
{
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);
}

inline void rpp_load48_u8pln3_to_u8pln3(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m128i *px)
{
    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);
}

inline void rpp_store48_u8pln3_to_u8pkd3(Rpp8u *dstPtr, __m128i *px)
{
    __m128i pxDst[4];
    __m128i pxZero = _mm_setzero_si128();
    __m128i pxMaskRGBAtoRGB = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
    pxDst[0] = _mm_unpacklo_epi8(px[1], pxZero);
    pxDst[1] = _mm_unpackhi_epi8(px[1], pxZero);
    pxDst[2] = _mm_unpacklo_epi8(px[0], px[2]);
    pxDst[3] = _mm_unpackhi_epi8(px[0], px[2]);
    _mm_storeu_si128((__m128i *)dstPtr, _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));
    _mm_storeu_si128((__m128i *)(dstPtr + 12), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));
    _mm_storeu_si128((__m128i *)(dstPtr + 24), _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));
    _mm_storeu_si128((__m128i *)(dstPtr + 36), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));
}


inline void compute_roi_validation_host(RpptROIPtr roiPtr, int srcW, int srcH)
{
    roiPtr->xywhROI.xy.x = max(0, min(roiPtr->xywhROI.xy.x, srcW - 1));
    roiPtr->xywhROI.xy.y = max(0, min(roiPtr->xywhROI.xy.y, srcH - 1));
    roiPtr->xywhROI.roiWidth = min(roiPtr->xywhROI.roiWidth, srcW - roiPtr->xywhROI.xy.x);
    roiPtr->xywhROI.roiHeight = min(roiPtr->xywhROI.roiHeight, srcH - roiPtr->xywhROI.xy.y);
}



RppStatus crop_u8_u8_host(Rpp8u *srcPtr, RpptDescPtr srcDescPtr, Rpp8u *dstPtr, RpptDescPtr dstDescPtr,
                          RpptROIPtr roiPtr, RpptRoiType roiType, RppLayoutParams layoutParams)
{
    RpptROI roi = *roiPtr;
    compute_roi_validation_host(&roi, srcDescPtr->w, srcDescPtr->h);

    Rpp8u *srcPtrImage = srcPtr;
    Rpp8u *dstPtrImage = dstPtr;
    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    // Crop without layout toggle (NHWC->NHWC or NCHW->NCHW)
    for(int c = 0; c < layoutParams.channelParam; c++)
    {
        Rpp8u *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrChannel;
        dstPtrRow = dstPtrChannel;

        for(int i = 0; i < roi.xywhROI.roiHeight; i++)
        {
            memcpy(dstPtrRow, srcPtrRow, bufferLength);
            srcPtrRow += srcDescPtr->strides.hStride;
            dstPtrRow += dstDescPtr->strides.hStride;
        }

        srcPtrChannel += srcDescPtr->strides.cStride;
        dstPtrChannel += dstDescPtr->strides.cStride;
    }

    return RPP_SUCCESS;
}


vector<Mat> loadBatchImages(const string& directory, int& batchSize, int& maxWidth, int& maxHeight, bool isColor)
{
    vector<Mat> images;
    DIR* dir;
    struct dirent* entry;

    maxWidth = 0;
    maxHeight = 0;

    if ((dir = opendir(directory.c_str())) == NULL) {
        cerr << "Could not open directory: " << directory << endl;
        return images;
    }

    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;

        if (filename == "." || filename == "..") continue;

        string ext = filename.substr(filename.find_last_of(".") + 1);
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != "jpg" && ext != "jpeg" && ext != "png" && ext != "bmp" && ext != "tiff")
            continue;

        string filePath = directory + "/" + filename;
        Mat img = imread(filePath, isColor ? IMREAD_COLOR : IMREAD_GRAYSCALE);

        if (!img.empty()) {
            maxWidth = max(maxWidth, img.cols);
            maxHeight = max(maxHeight, img.rows);
            images.push_back(move(img));
        } else {
            cerr << "Warning: Could not read image " << filePath << endl;
        }
    }

    closedir(dir);
    batchSize = images.size();
    return images;
}

void convert_nhwc_to_nchw(const vector<Mat>& nhwc_imgs, vector<vector<Rpp8u>>& nchw_imgs)
{
    nchw_imgs.resize(nhwc_imgs.size());
    
    for (size_t i = 0; i < nhwc_imgs.size(); i++) {
        const Mat& img = nhwc_imgs[i];
        int w = img.cols;
        int h = img.rows;
        int channels = img.channels();
        
        nchw_imgs[i].resize(w * h * channels);
        int channelSize = w * h;
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int nhwc_idx = (y * w + x) * channels;
                int nchw_idx = y * w + x;
                
                const Rpp8u* pixel = img.data + nhwc_idx;
                for (int c = 0; c < channels; c++) {
                    nchw_imgs[i][nchw_idx + c * channelSize] = pixel[c];
                }
            }
        }
    }
}

void benchmarkOpenCV_Crop(const vector<Mat>& imgs, int cropW, int cropH)
{
    int batchSize = imgs.size();
    if (batchSize == 0) {
        cerr << "Error: No images to process!" << endl;
        return;
    }

    // Filter images that are large enough
    vector<int> validIndices;
    for (int i = 0; i < batchSize; i++) {
        if (imgs[i].cols >= cropW && imgs[i].rows >= cropH) {
            validIndices.push_back(i);
        }
    }
    
    if (validIndices.empty()) {
        cerr << "Error: No images large enough for " << cropW << "x" << cropH << " crop!" << endl;
        return;
    }

    vector<Mat> output(validIndices.size());
    
    // Define crop ROI (center crop)
    vector<Rect> rois(validIndices.size());
    for (size_t idx = 0; idx < validIndices.size(); idx++) {
        int i = validIndices[idx];
        int x = (imgs[i].cols - cropW) / 2;
        int y = (imgs[i].rows - cropH) / 2;
        rois[idx] = Rect(x, y, cropW, cropH);
    }

    auto start = high_resolution_clock::now();

    // Run benchmark
    for (int k = 0; k < NUM_ITERATIONS; k++) {
        for (size_t idx = 0; idx < validIndices.size(); idx++) {
            int i = validIndices[idx];
            output[idx] = imgs[i](rois[idx]).clone();
        }
    }

    auto end = high_resolution_clock::now();

    double avg_time = duration_cast<microseconds>(end - start).count() / (NUM_ITERATIONS * 1000.0);
    double fps = (NUM_ITERATIONS * validIndices.size() * 1000.0) / duration_cast<milliseconds>(end - start).count();
    
    cout << "OpenCV Crop          : " << fixed << setprecision(3) << setw(8) << avg_time << " ms  |  " 
         << setw(8) << setprecision(1) << fps << " FPS (" << validIndices.size() << " imgs)" << endl;
}

void benchmarkRPP_Crop_NHWC(const vector<Mat>& imgs, int cropW, int cropH)
{
    int batchSize = imgs.size();
    if (batchSize == 0) {
        cerr << "Error: No images to process!" << endl;
        return;
    }

    // Filter images that are large enough
    vector<int> validIndices;
    for (int i = 0; i < batchSize; i++) {
        if (imgs[i].cols >= cropW && imgs[i].rows >= cropH) {
            validIndices.push_back(i);
        }
    }
    
    if (validIndices.empty()) {
        cerr << "Error: No images large enough for " << cropW << "x" << cropH << " crop!" << endl;
        return;
    }

    int channels = 3;
    
    // Allocate output images
    vector<vector<Rpp8u>> output(validIndices.size());
    for (size_t idx = 0; idx < validIndices.size(); idx++)
        output[idx].resize(cropW * cropH * channels);

    // Initialize descriptors and ROIs
    vector<RpptDesc> srcDescs(validIndices.size()), dstDescs(validIndices.size());
    vector<RpptROI> rois(validIndices.size());
    
    for (size_t idx = 0; idx < validIndices.size(); idx++) {
        int i = validIndices[idx];
        const Mat& img = imgs[i];
        
        // Set ROI (center crop)
        rois[idx].xywhROI.xy.x = (img.cols - cropW) / 2;
        rois[idx].xywhROI.xy.y = (img.rows - cropH) / 2;
        rois[idx].xywhROI.roiWidth = cropW;
        rois[idx].xywhROI.roiHeight = cropH;
        
        // Source descriptor (NHWC)
        srcDescs[idx].h = img.rows;
        srcDescs[idx].w = img.cols;
        srcDescs[idx].c = channels;
        srcDescs[idx].layout = RpptLayout::NHWC;
        srcDescs[idx].strides.hStride = img.cols * channels;
        srcDescs[idx].strides.wStride = channels;
        srcDescs[idx].strides.cStride = 1;
        
        // Destination descriptor (NHWC)
        dstDescs[idx].h = cropH;
        dstDescs[idx].w = cropW;
        dstDescs[idx].c = channels;
        dstDescs[idx].layout = RpptLayout::NHWC;
        dstDescs[idx].strides.hStride = cropW * channels;
        dstDescs[idx].strides.wStride = channels;
        dstDescs[idx].strides.cStride = 1;
    }

    RppLayoutParams layoutParams = {(Rpp32u)channels, (Rpp32u)channels};

    auto start = high_resolution_clock::now();

    // Run benchmark
    for (int k = 0; k < NUM_ITERATIONS; k++) {
        for (size_t idx = 0; idx < validIndices.size(); idx++) {
            int i = validIndices[idx];
            crop_u8_u8_host(imgs[i].data, &srcDescs[idx], output[idx].data(), &dstDescs[idx],
                           &rois[idx], RpptRoiType::XYWH, layoutParams);
        }
    }

    auto end = high_resolution_clock::now();

    double avg_time = duration_cast<microseconds>(end - start).count() / (NUM_ITERATIONS * 1000.0);
    double fps = (NUM_ITERATIONS * validIndices.size() * 1000.0) / duration_cast<milliseconds>(end - start).count();
    
    cout << "RPP Crop NHWC->NHWC  : " << fixed << setprecision(3) << setw(8) << avg_time << " ms  |  " 
         << setw(8) << setprecision(1) << fps << " FPS (" << validIndices.size() << " imgs)" << endl;
}

void benchmarkRPP_Crop_NCHW(const vector<vector<Rpp8u>>& nchw_imgs, const vector<Mat>& imgs, int cropW, int cropH)
{
    int batchSize = imgs.size();
    if (batchSize == 0) {
        cerr << "Error: No images to process!" << endl;
        return;
    }

    // Filter images that are large enough
    vector<int> validIndices;
    for (int i = 0; i < batchSize; i++) {
        if (imgs[i].cols >= cropW && imgs[i].rows >= cropH) {
            validIndices.push_back(i);
        }
    }
    
    if (validIndices.empty()) {
        cerr << "Error: No images large enough for " << cropW << "x" << cropH << " crop!" << endl;
        return;
    }

    int channels = 3;
    
    // Allocate output images
    vector<vector<Rpp8u>> output(validIndices.size());
    for (size_t idx = 0; idx < validIndices.size(); idx++)
        output[idx].resize(cropW * cropH * channels);

    // Initialize descriptors and ROIs
    vector<RpptDesc> srcDescs(validIndices.size()), dstDescs(validIndices.size());
    vector<RpptROI> rois(validIndices.size());
    
    for (size_t idx = 0; idx < validIndices.size(); idx++) {
        int i = validIndices[idx];
        const Mat& img = imgs[i];
        
        // Set ROI (center crop)
        rois[idx].xywhROI.xy.x = (img.cols - cropW) / 2;
        rois[idx].xywhROI.xy.y = (img.rows - cropH) / 2;
        rois[idx].xywhROI.roiWidth = cropW;
        rois[idx].xywhROI.roiHeight = cropH;
        
        // Source descriptor (NCHW)
        srcDescs[idx].h = img.rows;
        srcDescs[idx].w = img.cols;
        srcDescs[idx].c = channels;
        srcDescs[idx].layout = RpptLayout::NCHW;
        srcDescs[idx].strides.hStride = img.cols;
        srcDescs[idx].strides.wStride = 1;
        srcDescs[idx].strides.cStride = img.cols * img.rows;
        
        // Destination descriptor (NCHW)
        dstDescs[idx].h = cropH;
        dstDescs[idx].w = cropW;
        dstDescs[idx].c = channels;
        dstDescs[idx].layout = RpptLayout::NCHW;
        dstDescs[idx].strides.hStride = cropW;
        dstDescs[idx].strides.wStride = 1;
        dstDescs[idx].strides.cStride = cropW * cropH;
    }

    RppLayoutParams layoutParams = {1, (Rpp32u)channels};

    auto start = high_resolution_clock::now();

    // Run benchmark
    for (int k = 0; k < NUM_ITERATIONS; k++) {
        for (size_t idx = 0; idx < validIndices.size(); idx++) {
            int i = validIndices[idx];
            crop_u8_u8_host((Rpp8u*)nchw_imgs[i].data(), &srcDescs[idx], output[idx].data(), &dstDescs[idx],
                           &rois[idx], RpptRoiType::XYWH, layoutParams);
        }
    }

    auto end = high_resolution_clock::now();

    double avg_time = duration_cast<microseconds>(end - start).count() / (NUM_ITERATIONS * 1000.0);
    double fps = (NUM_ITERATIONS * validIndices.size() * 1000.0) / duration_cast<milliseconds>(end - start).count();
    
    cout << "RPP Crop NCHW->NCHW  : " << fixed << setprecision(3) << setw(8) << avg_time << " ms  |  " 
         << setw(8) << setprecision(1) << fps << " FPS (" << validIndices.size() << " imgs)" << endl;
}



int main(int argc, char** argv) {
    string imageDir = (argc > 1) ? argv[1] : ".";
    
    cout << "\n===============================================================" << endl;
    cout << "  Crop Performance Benchmark: OpenCV vs RPP (Embedded)" << endl;
    cout << "  No RPP Library Required - Standalone Code!" << endl;
    cout << "===============================================================\n" << endl;

    // Load images
    int batchSize = 0, maxWidth = 0, maxHeight = 0;
    cout << "Loading images from: " << imageDir << endl;
    vector<Mat> imgs = loadBatchImages(imageDir, batchSize, maxWidth, maxHeight, true);

    if (imgs.empty()) {
        cerr << "Error: No images found in directory!" << endl;
        return -1;
    }

    cout << "Loaded " << batchSize << " images" << endl;
    cout << "Max dimensions: " << maxWidth << "x" << maxHeight << endl;
    cout << "Channels: " << imgs[0].channels() << endl;
    cout << "Iterations: " << NUM_ITERATIONS << "\n" << endl;

    // Convert to NCHW format
    cout << "Converting images to NCHW format..." << endl;
    vector<vector<Rpp8u>> nchw_imgs;
    convert_nhwc_to_nchw(imgs, nchw_imgs);
    cout << "Conversion complete\n" << endl;

    // Define crop sizes to test
    vector<pair<int, int>> cropSizes = {
        {224, 224},  // Common CNN input size
        {512, 512},  // Medium crop
        {1024, 1024} // Large crop
    };

    for (const auto& cropSize : cropSizes) {
        int cropW = min(cropSize.first, maxWidth);
        int cropH = min(cropSize.second, maxHeight);
        
        if (cropW < 64 || cropH < 64) continue; 
        
        cout << "---------------------------------------------------------------" << endl;
        cout << "  Crop Size: " << cropW << "x" << cropH << " | Batch: " << batchSize << " images" << endl;
        cout << "---------------------------------------------------------------" << endl;
        
        benchmarkOpenCV_Crop(imgs, cropW, cropH);
        benchmarkRPP_Crop_NHWC(imgs, cropW, cropH);
        benchmarkRPP_Crop_NCHW(nchw_imgs, imgs, cropW, cropH);
        
        cout << endl;
    }

    cout << "===============================================================" << endl;
    cout << "  Benchmark Complete!" << endl;
    cout << "===============================================================\n" << endl;

    return 0;
}