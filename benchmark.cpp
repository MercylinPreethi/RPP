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

void convert_to_grayscale(const vector<Mat>& rgb_imgs, vector<Mat>& gray_imgs)
{
    gray_imgs.resize(rgb_imgs.size());
    for (size_t i = 0; i < rgb_imgs.size(); i++) {
        cvtColor(rgb_imgs[i], gray_imgs[i], COLOR_BGR2GRAY);
    }
}


struct ImageCropInfo {
    Rpp8u* srcPtr;
    int srcRowStep;
    int roiX;
    int roiY;
    bool isValid;
};

void benchmarkOpenCV_Crop_RGB(const vector<Mat>& imgs, int cropW, int cropH)
{
    int batchSize = imgs.size();
    if (batchSize == 0) return;

    vector<ImageCropInfo> cropInfo(batchSize);
    int validCount = 0;
    
    int channels = 3; 
    size_t pixelsPerCrop = (size_t)cropW * cropH * channels;
    vector<Rpp8u> flatOutput(batchSize * pixelsPerCrop);

    for (int i = 0; i < batchSize; i++) {
        cropInfo[i].srcPtr = imgs[i].data;
        cropInfo[i].srcRowStep = imgs[i].step[0];
        
        if (imgs[i].cols >= cropW && imgs[i].rows >= cropH) {
            cropInfo[i].roiX = (imgs[i].cols - cropW) / 2;
            cropInfo[i].roiY = (imgs[i].rows - cropH) / 2;
            cropInfo[i].isValid = true;
            validCount++;
        } else {
            cropInfo[i].isValid = false;
        }
    }
    
    if (validCount == 0) {
        cout << "OpenCV RGB (3-ch)    : SKIPPED (no images >= " << cropW << "x" << cropH << ")" << endl;
        return;
    }

    microseconds total_duration(0);
    int total_valid_ops = 0;

    for (int k = 0; k < NUM_ITERATIONS; k++) {
        for (int i = 0; i < batchSize; i++) {
            const ImageCropInfo& info = cropInfo[i];  // Single struct access
            
            if (info.isValid) {
                int bytesToCopyPerRow = cropW * channels;
                Rpp8u* src = info.srcPtr + (info.roiY * info.srcRowStep) + (info.roiX * channels);
                Rpp8u* dst = flatOutput.data() + (i * pixelsPerCrop);

                auto start = high_resolution_clock::now();

                for (int r = 0; r < cropH; r++) {
                    memcpy(dst + (r * bytesToCopyPerRow), 
                           src + (r * info.srcRowStep), 
                           bytesToCopyPerRow);
                }

                auto end = high_resolution_clock::now();

                total_duration += duration_cast<microseconds>(end - start);
                total_valid_ops++;
            }
        }
    }

    double avg_time_ms = (total_duration.count() / 1000.0) / total_valid_ops;
    
    cout << "OpenCV RGB (3-ch)    : " << fixed << setprecision(3) << setw(8) << avg_time_ms << " ms  |  " 
         << setw(8) << setprecision(1) << endl;
}

void benchmarkOpenCV_Crop_Grayscale(const vector<Mat>& gray_imgs, int cropW, int cropH)
{
    int batchSize = gray_imgs.size();
    if (batchSize == 0) return;

    struct GrayCropInfo {
        Rpp8u* srcPtr;
        int srcCols;
        int roiX;
        int roiY;
        bool isValid;
    };
    
    vector<GrayCropInfo> cropInfo(batchSize);
    int validCount = 0;
    
    int channels = 1; 
    vector<Rpp8u> flatOutput(batchSize * cropW * cropH);

    // Initialize everything in one pass
    for (int i = 0; i < batchSize; i++) {
        cropInfo[i].srcPtr = gray_imgs[i].data;
        cropInfo[i].srcCols = gray_imgs[i].cols;
        
        if (gray_imgs[i].cols >= cropW && gray_imgs[i].rows >= cropH) {
            cropInfo[i].roiX = (gray_imgs[i].cols - cropW) / 2;
            cropInfo[i].roiY = (gray_imgs[i].rows - cropH) / 2;
            cropInfo[i].isValid = true;
            validCount++;
        } else {
            cropInfo[i].isValid = false;
        }
    }
    
    if (validCount == 0) {
        cout << "OpenCV Gray (1-ch)   : SKIPPED (no images >= " << cropW << "x" << cropH << ")" << endl;
        return;
    }

    microseconds total_duration(0);
    int total_valid_calls = 0;

    for (int k = 0; k < NUM_ITERATIONS; k++) {
        for (int i = 0; i < batchSize; i++) {
            const GrayCropInfo& info = cropInfo[i];  // Single struct access
            
            if (info.isValid) {
                Rpp8u* src = info.srcPtr + (info.roiY * info.srcCols) + info.roiX;
                Rpp8u* dst = flatOutput.data() + (i * cropW * cropH);

                auto start = high_resolution_clock::now();

                for (int r = 0; r < cropH; r++) {
                    memcpy(dst + (r * cropW), src + (r * info.srcCols), cropW);
                }

                auto end = high_resolution_clock::now();

                total_duration += duration_cast<microseconds>(end - start);
                total_valid_calls++;
            }
        }
    }

    double avg_time_ms = (total_duration.count() / 1000.0) / total_valid_calls;

    cout << "OpenCV Gray (1-ch)   : " << fixed << setprecision(3) << setw(8) << avg_time_ms  << " ms  |  " 
         << setw(8) << setprecision(1) << endl;
}

void benchmarkRPP_Crop_NHWC(const vector<Mat>& imgs, int cropW, int cropH)
{
    int batchSize = imgs.size();
    if (batchSize == 0) return;

    int validCount = 0;
    for (int i = 0; i < batchSize; i++) {
        if (imgs[i].cols >= cropW && imgs[i].rows >= cropH) {
            validCount++;
        }
    }
    
    if (validCount == 0) {
        cout << "RPP NHWC (3-ch)      : SKIPPED (no images >= " << cropW << "x" << cropH << ")" << endl;
        return;
    }

    int channels = 3;
    

    vector<vector<Rpp8u>> output(batchSize);
    for (int i = 0; i < batchSize; i++)
        output[i].resize(cropW * cropH * channels);

    vector<RpptDesc> srcDescs(batchSize), dstDescs(batchSize);
    vector<RpptROI> rois(batchSize);
    vector<bool> isValid(batchSize);
    
    for (int i = 0; i < batchSize; i++) {
        if (imgs[i].cols >= cropW && imgs[i].rows >= cropH) {
            const Mat& img = imgs[i];
            
            rois[i].xywhROI.xy.x = (img.cols - cropW) / 2;
            rois[i].xywhROI.xy.y = (img.rows - cropH) / 2;
            rois[i].xywhROI.roiWidth = cropW;
            rois[i].xywhROI.roiHeight = cropH;
            
            srcDescs[i].h = img.rows;
            srcDescs[i].w = img.cols;
            srcDescs[i].c = channels;
            srcDescs[i].layout = RpptLayout::NHWC;
            srcDescs[i].strides.hStride = img.cols * channels;
            srcDescs[i].strides.wStride = channels;
            srcDescs[i].strides.cStride = 1;
            
            dstDescs[i].h = cropH;
            dstDescs[i].w = cropW;
            dstDescs[i].c = channels;
            dstDescs[i].layout = RpptLayout::NHWC;
            dstDescs[i].strides.hStride = cropW * channels;
            dstDescs[i].strides.wStride = channels;
            dstDescs[i].strides.cStride = 1;
            
            isValid[i] = true;
        } else {
            isValid[i] = false;
        }
    }

    RppLayoutParams layoutParams = {3, 1};

    microseconds total_duration(0);
    int total_valid_ops = 0;

    for (int k = 0; k < NUM_ITERATIONS; k++) {
        for (int i = 0; i < batchSize; i++) {
            if (isValid[i]) {

                auto start = high_resolution_clock::now();

                crop_u8_u8_host(imgs[i].data, &srcDescs[i], output[i].data(), &dstDescs[i],
                                &rois[i], RpptRoiType::XYWH, layoutParams);

                auto end = high_resolution_clock::now();
                total_duration += duration_cast<microseconds>(end - start);
                total_valid_ops++;
            }
        }
    }

    double avg_time_ms = (total_duration.count() / 1000.0) / total_valid_ops;
    
    cout << "RPP NHWC (3-ch)      : " << fixed << setprecision(3) << setw(8) << avg_time_ms << " ms  |  " 
         << setw(8) << setprecision(1) << endl;

}

void benchmarkRPP_Crop_NCHW(const vector<vector<Rpp8u>>& nchw_imgs, const vector<Mat>& imgs, int cropW, int cropH)
{
    int batchSize = imgs.size();
    if (batchSize == 0) return;

    int validCount = 0;
    for (int i = 0; i < batchSize; i++) {
        if (imgs[i].cols >= cropW && imgs[i].rows >= cropH) {
            validCount++;
        }
    }
    
    if (validCount == 0) {
        cout << "RPP NCHW (1-ch×3)    : SKIPPED (no images >= " << cropW << "x" << cropH << ")" << endl;
        return;
    }

    int channels = 3;
    
    
    vector<vector<Rpp8u>> output(batchSize);
    for (int i = 0; i < batchSize; i++)
        output[i].resize(cropW * cropH * channels);

    vector<RpptDesc> srcDescs(batchSize), dstDescs(batchSize);
    vector<RpptROI> rois(batchSize);
    vector<bool> isValid(batchSize);

    for (int i = 0; i < batchSize; i++) {
        if (imgs[i].cols >= cropW && imgs[i].rows >= cropH) {
            const Mat& img = imgs[i];
            
            rois[i].xywhROI.xy.x = (img.cols - cropW) / 2;
            rois[i].xywhROI.xy.y = (img.rows - cropH) / 2;
            rois[i].xywhROI.roiWidth = cropW;
            rois[i].xywhROI.roiHeight = cropH;
            
            srcDescs[i].h = img.rows;
            srcDescs[i].w = img.cols;
            srcDescs[i].c = channels;
            srcDescs[i].layout = RpptLayout::NCHW;
            srcDescs[i].strides.hStride = img.cols;
            srcDescs[i].strides.wStride = 1;
            srcDescs[i].strides.cStride = img.cols * img.rows;
            
            dstDescs[i].h = cropH;
            dstDescs[i].w = cropW;
            dstDescs[i].c = channels;
            dstDescs[i].layout = RpptLayout::NCHW;
            dstDescs[i].strides.hStride = cropW;
            dstDescs[i].strides.wStride = 1;
            dstDescs[i].strides.cStride = cropW * cropH;
            
            isValid[i] = true;
        } else {
            isValid[i] = false;
        }
    }

    RppLayoutParams layoutParams = {1, 1};

    microseconds total_duration(0);
    int total_valid_ops = 0;

    for (int k = 0; k < NUM_ITERATIONS; k++) {
        for (int i = 0; i < batchSize; i++) {
            if (isValid[i]) {
                
                for (int c = 0; c < channels; c++) {

                    Rpp8u* srcChannel = (Rpp8u*)nchw_imgs[i].data() + c * srcDescs[i].strides.cStride;
                    Rpp8u* dstChannel = output[i].data() + c * dstDescs[i].strides.cStride;

                    auto start = high_resolution_clock::now();    
                    
                    crop_u8_u8_host(srcChannel, &srcDescs[i], dstChannel, &dstDescs[i],
                                   &rois[i], RpptRoiType::XYWH, layoutParams);
                    
                    auto end = high_resolution_clock::now();

                    total_duration += duration_cast<microseconds>(end - start);
                    total_valid_ops++;
                }
            }
        }
    }

    double avg_time_ms = (total_duration.count() / 1000.0) / total_valid_ops;
    
    cout << "RPP NCHW (1-ch×3)    : " << fixed << setprecision(3) << setw(8) << avg_time_ms << " ms  |  " 
         << setw(8) << setprecision(1) << endl;
}

void benchmarkRPP_Crop_Grayscale(const vector<vector<Rpp8u>>& gray_nchw_imgs, 
                                  const vector<Mat>& gray_imgs, 
                                  int cropW, int cropH)
{
    int batchSize = gray_imgs.size();
    if (batchSize == 0) return;

    int validCount = 0;
    for (int i = 0; i < batchSize; i++) {
        if (gray_imgs[i].cols >= cropW && gray_imgs[i].rows >= cropH) {
            validCount++;
        }
    }
    
    if (validCount == 0) {
        cout << "RPP Gray (1-ch)      : SKIPPED (no images >= " << cropW << "x" << cropH << ")" << endl;
        return;
    }

    int channels = 1;


    vector<vector<Rpp8u>> output(batchSize);
    for (int i = 0; i < batchSize; i++)
        output[i].resize(cropW * cropH * channels);

    vector<RpptDesc> srcDescs(batchSize), dstDescs(batchSize);
    vector<RpptROI> rois(batchSize);
    vector<bool> isValid(batchSize);

    for (int i = 0; i < batchSize; i++) {
        if (gray_imgs[i].cols >= cropW && gray_imgs[i].rows >= cropH) {
            const Mat& img = gray_imgs[i];
            
            rois[i].xywhROI.xy.x = (img.cols - cropW) / 2;
            rois[i].xywhROI.xy.y = (img.rows - cropH) / 2;
            rois[i].xywhROI.roiWidth = cropW;
            rois[i].xywhROI.roiHeight = cropH;
            
            srcDescs[i].h = img.rows;
            srcDescs[i].w = img.cols;
            srcDescs[i].c = channels;
            srcDescs[i].layout = RpptLayout::NCHW;
            srcDescs[i].strides.hStride = img.cols;
            srcDescs[i].strides.wStride = 1;
            srcDescs[i].strides.cStride = img.cols * img.rows;
            
            dstDescs[i].h = cropH;
            dstDescs[i].w = cropW;
            dstDescs[i].c = channels;
            dstDescs[i].layout = RpptLayout::NCHW;
            dstDescs[i].strides.hStride = cropW;
            dstDescs[i].strides.wStride = 1;
            dstDescs[i].strides.cStride = cropW * cropH;
            
            isValid[i] = true;
        } else {
            isValid[i] = false;
        }
    }

    RppLayoutParams layoutParams = {1, 1};

    microseconds total_duration(0);
    int total_valid_ops = 0;

    for (int k = 0; k < NUM_ITERATIONS; k++) {
        for (int i = 0; i < batchSize; i++) {
            if (isValid[i]) {
                auto start = high_resolution_clock::now();
                
                crop_u8_u8_host((Rpp8u*)gray_nchw_imgs[i].data(), &srcDescs[i], 
                               output[i].data(), &dstDescs[i],
                               &rois[i], RpptRoiType::XYWH, layoutParams);
                
                auto end = high_resolution_clock::now();
                total_duration += duration_cast<microseconds>(end - start);
                total_valid_ops++;
            }
        }
    }

    double avg_time_ms = (total_duration.count() / 1000.0) / total_valid_ops;
    
    cout << "RPP Gray (1-ch)      : " << fixed << setprecision(3) << setw(8) << avg_time_ms << " ms  |  " 
         << setw(8) << setprecision(1) << endl;
}


int main(int argc, char** argv) {
    string imageDir = (argc > 1) ? argv[1] : ".";
    
    cout << "\n===============================================================" << endl;
    cout << "  Crop Performance Benchmark: OpenCV vs RPP" << endl;
    cout << "  Size-Safe Benchmarking (skips images that are too small)" << endl;
    cout << "===============================================================\n" << endl;

    int batchSize = 0, maxWidth = 0, maxHeight = 0;
    cout << "Loading RGB images from: " << imageDir << endl;
    vector<Mat> imgs_rgb = loadBatchImages(imageDir, batchSize, maxWidth, maxHeight, true);

    if (imgs_rgb.empty()) {
        cerr << "Error: No images found!" << endl;
        return -1;
    }

    cout << "Loaded " << batchSize << " RGB images" << endl;
    cout << "Max dimensions: " << maxWidth << "x" << maxHeight << endl;
    cout << "Iterations: " << NUM_ITERATIONS << "\n" << endl;

    cout << "Converting to Grayscale..." << endl;
    vector<Mat> imgs_gray;
    convert_to_grayscale(imgs_rgb, imgs_gray);
    cout << "Grayscale conversion complete\n" << endl;

    cout << "Converting RGB to NCHW format..." << endl;

    vector<vector<Rpp8u>> nchw_imgs_rgb;
    convert_nhwc_to_nchw(imgs_rgb, nchw_imgs_rgb);
    cout << "RGB NCHW conversion complete\n" << endl;

    cout << "Converting Grayscale to NCHW format..." << endl;
    vector<vector<Rpp8u>> nchw_imgs_gray;
    nchw_imgs_gray.resize(imgs_gray.size());
    for (size_t i = 0; i < imgs_gray.size(); i++) {
        int w = imgs_gray[i].cols;
        int h = imgs_gray[i].rows;
        nchw_imgs_gray[i].resize(w * h);
        memcpy(nchw_imgs_gray[i].data(), imgs_gray[i].data, w * h);
    }
    cout << "Grayscale NCHW conversion complete\n" << endl;

    vector<pair<int, int>> cropSizes = {
        {224, 224},
        {512, 512},
        {1024, 1024}
    };

    for (const auto& cropSize : cropSizes) {
        int cropW = min(cropSize.first, maxWidth);
        int cropH = min(cropSize.second, maxHeight);
        
        if (cropW < 64 || cropH < 64) continue;
        
        cout << "===============================================================" << endl;
        cout << "  Crop: " << cropW << "x" << cropH << " (" << (cropW*cropH)/1000.0 << "K px)  |  Batch: " << batchSize << " images" << endl;
        cout << "===============================================================" << endl;
        
        cout << "\n--- RGB (3-channel) Benchmarks ---" << endl;
        benchmarkOpenCV_Crop_RGB(imgs_rgb, cropW, cropH);
        benchmarkRPP_Crop_NHWC(imgs_rgb, cropW, cropH);
        benchmarkRPP_Crop_NCHW(nchw_imgs_rgb, imgs_rgb, cropW, cropH);
        
        cout << "\n--- Grayscale (1-channel) Benchmarks ---" << endl;
        benchmarkOpenCV_Crop_Grayscale(imgs_gray, cropW, cropH);
        benchmarkRPP_Crop_Grayscale(nchw_imgs_gray, imgs_gray, cropW, cropH);
        
        cout << endl;
    }

    cout << "===============================================================" << endl;
    cout << "  Benchmark Complete!" << endl;
    cout << "  Note: Only images >= crop size were benchmarked" << endl;
    cout << "===============================================================\n" << endl;

    return 0;
}