/*
 * Comprehensive Crop Demo - All Possible Crop Positions
 * Crops an image in multiple ways: center, corners, edges, margins
 * Saves each with descriptive filename
 * 
 * Compile: g++ -o crop_demo comprehensive_crop_demo.cpp -std=c++11 -O3 -msse4.2 -mavx2
 * Run: ./crop_demo input.jpg
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <string>
#include <iomanip>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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


inline void compute_roi_validation_host(RpptROIPtr roiPtr, RpptROI* roi, RpptROI* roiDefault, RpptRoiType roiType) {
    if (roiPtr) {
        *roi = *roiPtr;
    } else {
        *roi = *roiDefault;
    }
    
    if (roi->xywhROI.xy.x < 0) roi->xywhROI.xy.x = 0;
    if (roi->xywhROI.xy.y < 0) roi->xywhROI.xy.y = 0;
    if (roi->xywhROI.xy.x + roi->xywhROI.roiWidth > roiDefault->xywhROI.roiWidth)
        roi->xywhROI.roiWidth = roiDefault->xywhROI.roiWidth - roi->xywhROI.xy.x;
    if (roi->xywhROI.xy.y + roi->xywhROI.roiHeight > roiDefault->xywhROI.roiHeight)
        roi->xywhROI.roiHeight = roiDefault->xywhROI.roiHeight - roi->xywhROI.xy.y;
}


RppStatus crop_u8_u8_host_standalone(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiPtr,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    RpptROI roi;
    compute_roi_validation_host(roiPtr, &roi, &roiDefault, roiType);

    Rpp8u *srcPtrImage = srcPtr;
    Rpp8u *dstPtrImage = dstPtr;

    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    // Crop with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u alignedLength = (bufferLength / 48) * 48;

        Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        for(int i = 0; i < roi.xywhROI.roiHeight; i++)
        {
            Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow;
            dstPtrTempR = dstPtrRowR;
            dstPtrTempG = dstPtrRowG;
            dstPtrTempB = dstPtrRowB;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
            {
                __m128i px[3];
                rpp_load48_u8pkd3_to_u8pln3(srcPtrTemp, px);
                rpp_store48_u8pln3_to_u8pln3(dstPtrTempR, dstPtrTempG, dstPtrTempB, px);
                srcPtrTemp += 48;
                dstPtrTempR += 16;
                dstPtrTempG += 16;
                dstPtrTempB += 16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
            {
                *dstPtrTempR = srcPtrTemp[0];
                *dstPtrTempG = srcPtrTemp[1];
                *dstPtrTempB = srcPtrTemp[2];
                srcPtrTemp += 3;
                dstPtrTempR++;
                dstPtrTempG++;
                dstPtrTempB++;
            }

            srcPtrRow += srcDescPtr->strides.hStride;
            dstPtrRowR += dstDescPtr->strides.hStride;
            dstPtrRowG += dstDescPtr->strides.hStride;
            dstPtrRowB += dstDescPtr->strides.hStride;
        }
    }

    // Crop with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u alignedLength = (bufferLength / 48) * 48;

        Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

        for(int i = 0; i < roi.xywhROI.roiHeight; i++)
        {
            Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR;
            srcPtrTempG = srcPtrRowG;
            srcPtrTempB = srcPtrRowB;
            dstPtrTemp = dstPtrRow;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
            {
                __m128i px[3];
                rpp_load48_u8pln3_to_u8pln3(srcPtrTempR, srcPtrTempG, srcPtrTempB, px);
                rpp_store48_u8pln3_to_u8pkd3(dstPtrTemp, px);
                srcPtrTempR += 16;
                srcPtrTempG += 16;
                srcPtrTempB += 16;
                dstPtrTemp += 48;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                dstPtrTemp[0] = *srcPtrTempR;
                dstPtrTemp[1] = *srcPtrTempG;
                dstPtrTemp[2] = *srcPtrTempB;
                srcPtrTempR++;
                srcPtrTempG++;
                srcPtrTempB++;
                dstPtrTemp += 3;
            }

            srcPtrRowR += srcDescPtr->strides.hStride;
            srcPtrRowG += srcDescPtr->strides.hStride;
            srcPtrRowB += srcDescPtr->strides.hStride;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // Crop without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    else
    {
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
    }

    return RPP_SUCCESS;
}

struct CropConfig {
    std::string name;
    int x, y, width, height;
    std::string description;
};


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "\nUsage: " << argv[0] << " <input_image>\n" << std::endl;
        std::cout << "Example: " << argv[0] << " photo.jpg\n" << std::endl;
        return -1;
    }

    const char* input_path = argv[1];
    
    // Load image
    int srcW, srcH, srcC;
    unsigned char* pixels = stbi_load(input_path, &srcW, &srcH, &srcC, 3);
    if (!pixels) {
        std::cout << "Error: Could not load image " << input_path << std::endl;
        return -1;
    }

    std::cout << "\n════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Comprehensive Crop Demo - All Crop Positions" << std::endl;
    std::cout << "════════════════════════════════════════════════════════════════\n" << std::endl;
    std::cout << "Loaded Image: " << input_path << " (" << srcW << "x" << srcH << ")" << std::endl;

    std::vector<Rpp8u> srcImage(pixels, pixels + (srcW * srcH * 3));
    stbi_image_free(pixels);

    // Define crop size (adjust based on image size)
    int cropW = std::min(300, srcW / 2);
    int cropH = std::min(300, srcH / 2);

    // Define all crop configurations
    std::vector<CropConfig> crops;

    // 1. CENTER CROP
    crops.push_back({
        "center",
        (srcW - cropW) / 2,
        (srcH - cropH) / 2,
        cropW, cropH,
        "Center of image"
    });

    // 2. CORNER CROPS
    crops.push_back({
        "top_left",
        0, 0,
        cropW, cropH,
        "Top-left corner"
    });

    crops.push_back({
        "top_right",
        srcW - cropW, 0,
        cropW, cropH,
        "Top-right corner"
    });

    crops.push_back({
        "bottom_left",
        0, srcH - cropH,
        cropW, cropH,
        "Bottom-left corner"
    });

    crops.push_back({
        "bottom_right",
        srcW - cropW, srcH - cropH,
        cropW, cropH,
        "Bottom-right corner"
    });

    // 3. EDGE MIDPOINT CROPS
    crops.push_back({
        "top_center",
        (srcW - cropW) / 2, 0,
        cropW, cropH,
        "Top edge center"
    });

    crops.push_back({
        "bottom_center",
        (srcW - cropW) / 2, srcH - cropH,
        cropW, cropH,
        "Bottom edge center"
    });

    crops.push_back({
        "left_center",
        0, (srcH - cropH) / 2,
        cropW, cropH,
        "Left edge center"
    });

    crops.push_back({
        "right_center",
        srcW - cropW, (srcH - cropH) / 2,
        cropW, cropH,
        "Right edge center"
    });

    // 4. MARGIN CROPS (10% from edges)
    int margin = (int)(std::min(srcW, srcH) * 0.1);
    
    crops.push_back({
        "margin_top_left",
        margin, margin,
        cropW, cropH,
        "Top-left with 10% margin"
    });

    crops.push_back({
        "margin_top_right",
        srcW - cropW - margin, margin,
        cropW, cropH,
        "Top-right with 10% margin"
    });

    crops.push_back({
        "margin_bottom_left",
        margin, srcH - cropH - margin,
        cropW, cropH,
        "Bottom-left with 10% margin"
    });

    crops.push_back({
        "margin_bottom_right",
        srcW - cropW - margin, srcH - cropH - margin,
        cropW, cropH,
        "Bottom-right with 10% margin"
    });

    // 5. QUADRANT CROPS
    crops.push_back({
        "quadrant_1_top_left",
        srcW / 4 - cropW / 2, srcH / 4 - cropH / 2,
        cropW, cropH,
        "First quadrant (top-left)"
    });

    crops.push_back({
        "quadrant_2_top_right",
        3 * srcW / 4 - cropW / 2, srcH / 4 - cropH / 2,
        cropW, cropH,
        "Second quadrant (top-right)"
    });

    crops.push_back({
        "quadrant_3_bottom_left",
        srcW / 4 - cropW / 2, 3 * srcH / 4 - cropH / 2,
        cropW, cropH,
        "Third quadrant (bottom-left)"
    });

    crops.push_back({
        "quadrant_4_bottom_right",
        3 * srcW / 4 - cropW / 2, 3 * srcH / 4 - cropH / 2,
        cropW, cropH,
        "Fourth quadrant (bottom-right)"
    });

    // 6. DIFFERENT SIZE CROPS FROM CENTER
    crops.push_back({
        "center_small",
        (srcW - cropW/2) / 2, (srcH - cropH/2) / 2,
        cropW/2, cropH/2,
        "Center - small size (50%)"
    });

    crops.push_back({
        "center_large",
        (srcW - cropW*3/2) / 2, (srcH - cropH*3/2) / 2,
        std::min(cropW*3/2, srcW), std::min(cropH*3/2, srcH),
        "Center - large size (150%)"
    });

    // 7. FULL WIDTH/HEIGHT CROPS
    crops.push_back({
        "full_width_top",
        0, 0,
        srcW, cropH,
        "Full width from top"
    });

    crops.push_back({
        "full_width_center",
        0, (srcH - cropH) / 2,
        srcW, cropH,
        "Full width from center"
    });

    crops.push_back({
        "full_height_left",
        0, 0,
        cropW, srcH,
        "Full height from left"
    });

    crops.push_back({
        "full_height_center",
        (srcW - cropW) / 2, 0,
        cropW, srcH,
        "Full height from center"
    });

    // Setup source descriptor
    RpptDesc srcDesc = {0, 3, (Rpp32u)srcH, (Rpp32u)srcW, RpptLayout::NHWC, {0, (Rpp32u)srcW*3, 0, 1}};
    RppLayoutParams layoutParams = {3, 1};

    std::cout << "\nProcessing " << crops.size() << " different crop configurations...\n" << std::endl;

    int successCount = 0;
    
    // Process each crop
    for (const auto& crop : crops) {
        // Validate crop boundaries
        int actualX = std::max(0, std::min(crop.x, srcW - 1));
        int actualY = std::max(0, std::min(crop.y, srcH - 1));
        int actualW = std::min(crop.width, srcW - actualX);
        int actualH = std::min(crop.height, srcH - actualY);

        if (actualW <= 0 || actualH <= 0) {
            std::cout << "⚠ Skipping " << crop.name << " - invalid dimensions" << std::endl;
            continue;
        }

        // Allocate output buffer
        std::vector<Rpp8u> dstImage(actualW * actualH * 3);

        // Setup destination descriptor
        RpptDesc dstDesc = {0, 3, (Rpp32u)actualH, (Rpp32u)actualW, RpptLayout::NHWC, {0, (Rpp32u)actualW*3, 0, 1}};

        // Setup ROI
        RpptROI roi;
        roi.xywhROI.xy.x = actualX;
        roi.xywhROI.xy.y = actualY;
        roi.xywhROI.roiWidth = actualW;
        roi.xywhROI.roiHeight = actualH;

        // Perform crop
        RppStatus status = crop_u8_u8_host_standalone(
            srcImage.data(), &srcDesc,
            dstImage.data(), &dstDesc,
            &roi, RpptRoiType::XYWH,
            layoutParams
        );

        if (status == RPP_SUCCESS) {
            // Generate output filename
            std::string output_filename = "crop_" + crop.name + ".jpg";
            
            // Save image
            if (stbi_write_jpg(output_filename.c_str(), actualW, actualH, 3, dstImage.data(), 90)) {
                std::cout << "✓ " << std::left << std::setw(25) << crop.name 
                          << " → " << output_filename 
                          << "  [" << actualW << "×" << actualH << " from (" 
                          << actualX << "," << actualY << ")]" << std::endl;
                std::cout << "  " << crop.description << std::endl;
                successCount++;
            } else {
                std::cout << "✗ Failed to save " << output_filename << std::endl;
            }
        } else {
            std::cout << "✗ Crop failed for " << crop.name << std::endl;
        }
    }

    std::cout << "\n════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Completed: " << successCount << "/" << crops.size() << " crops saved successfully" << std::endl;
    std::cout << "════════════════════════════════════════════════════════════════\n" << std::endl;

    return 0;
}