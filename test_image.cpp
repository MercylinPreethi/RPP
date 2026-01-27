#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  // SSE4.1
#include <immintrin.h>  // AVX2

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef unsigned char Rpp8u;
typedef signed char Rpp8s;
typedef float Rpp32f;
typedef unsigned short Rpp16f;
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

// Global SIMD constants
static const __m128i xmm_pxConvertI8 = _mm_set1_epi8((char)128);


inline void compute_xywh_from_ltrb_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtrImage)
{
    roiPtrImage->xywhROI.xy.x = roiPtrInput->ltrbROI.lt.x;
    roiPtrImage->xywhROI.xy.y = roiPtrInput->ltrbROI.lt.y;
    roiPtrImage->xywhROI.roiWidth = roiPtrInput->ltrbROI.rb.x - roiPtrInput->ltrbROI.lt.x + 1;
    roiPtrImage->xywhROI.roiHeight = roiPtrInput->ltrbROI.rb.y - roiPtrInput->ltrbROI.lt.y + 1;
}


inline void compute_roi_boundary_check_host(RpptROIPtr roiPtrImage, RpptROIPtr roiPtr, RpptROIPtr roiPtrDefault)
{
    roiPtr->xywhROI.xy.x = std::max(roiPtrDefault->xywhROI.xy.x, roiPtrImage->xywhROI.xy.x);
    roiPtr->xywhROI.xy.y = std::max(roiPtrDefault->xywhROI.xy.y, roiPtrImage->xywhROI.xy.y);
    roiPtr->xywhROI.roiWidth = std::min(roiPtrDefault->xywhROI.roiWidth - roiPtrImage->xywhROI.xy.x, roiPtrImage->xywhROI.roiWidth);
    roiPtr->xywhROI.roiHeight = std::min(roiPtrDefault->xywhROI.roiHeight - roiPtrImage->xywhROI.xy.y, roiPtrImage->xywhROI.roiHeight);
}

inline void compute_roi_validation_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtr, RpptROIPtr roiPtrDefault, RpptRoiType roiType)
{
    if (roiPtrInput == NULL)
    {
        roiPtr = roiPtrDefault;
    }
    else
    {
        RpptROI roiImage;
        RpptROIPtr roiPtrImage = &roiImage;
        if (roiType == RpptRoiType::LTRB)
            compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
        else if (roiType == RpptRoiType::XYWH)
            roiPtrImage = roiPtrInput;
        compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
    }
}


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

inline void rpp_load12_f32pkd3_to_f32pln3(Rpp32f *srcPtr, __m128 *p)
{
    p[0] = _mm_loadu_ps(srcPtr);
    p[1] = _mm_loadu_ps(srcPtr + 3);
    p[2] = _mm_loadu_ps(srcPtr + 6);
    p[3] = _mm_loadu_ps(srcPtr + 9);
    _MM_TRANSPOSE4_PS(p[0], p[1], p[2], p[3]);
}

inline void rpp_store12_f32pln3_to_f32pln3(Rpp32f *dstPtrR, Rpp32f *dstPtrG, Rpp32f *dstPtrB, __m128 *p)
{
    _mm_storeu_ps(dstPtrR, p[0]);
    _mm_storeu_ps(dstPtrG, p[1]);
    _mm_storeu_ps(dstPtrB, p[2]);
}

inline void rpp_load12_f32pln3_to_f32pln3(Rpp32f *srcPtrR, Rpp32f *srcPtrG, Rpp32f *srcPtrB, __m128 *p)
{
    p[0] = _mm_loadu_ps(srcPtrR);
    p[1] = _mm_loadu_ps(srcPtrG);
    p[2] = _mm_loadu_ps(srcPtrB);
}

inline void rpp_store12_f32pln3_to_f32pkd3(Rpp32f *dstPtr, __m128 *p)
{
    _MM_TRANSPOSE4_PS(p[0], p[1], p[2], p[3]);
    _mm_storeu_ps(dstPtr, p[0]);
    _mm_storeu_ps(dstPtr + 3, p[1]);
    _mm_storeu_ps(dstPtr + 6, p[2]);
    _mm_storeu_ps(dstPtr + 9, p[3]);
}

inline void rpp_load24_f16pkd3_to_f32pln3_avx(Rpp16f *srcPtr, __m256 *p)
{
    __m128 p128[8];
    p128[0] = _mm_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr))));
    p128[1] = _mm_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 3))));
    p128[2] = _mm_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 6))));
    p128[3] = _mm_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 9))));
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    p128[4] = _mm_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 12))));
    p128[5] = _mm_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 15))));
    p128[6] = _mm_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 18))));
    p128[7] = _mm_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 21))));
    _MM_TRANSPOSE4_PS(p128[4], p128[5], p128[6], p128[7]);
    p[0] = _mm256_setr_m128(p128[0], p128[4]);
    p[1] = _mm256_setr_m128(p128[1], p128[5]);
    p[2] = _mm256_setr_m128(p128[2], p128[6]);
}

inline void rpp_store24_f32pln3_to_f16pkd3_avx(Rpp16f* dstPtr, __m256* p)
{
    __m128 p128[4];
    p128[0] = _mm256_extractf128_ps(p[0], 0);
    p128[1] = _mm256_extractf128_ps(p[1], 0);
    p128[2] = _mm256_extractf128_ps(p[2], 0);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);

    __m128i px128[4];
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstPtr, px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 3), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 6), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 9), px128[3]);

    p128[0] = _mm256_extractf128_ps(p[0], 1);
    p128[1] = _mm256_extractf128_ps(p[1], 1);
    p128[2] = _mm256_extractf128_ps(p[2], 1);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);

    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 15), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 18), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 21), px128[3]);
}


inline void rpp_store24_f32pln3_to_f16pln3_avx(Rpp16f* dstRPtr, Rpp16f* dstGPtr, Rpp16f* dstBPtr, __m256* p)
{
    __m128i px128[3];
    px128[0] = _mm256_cvtps_ph(p[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm256_cvtps_ph(p[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm256_cvtps_ph(p[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstRPtr, px128[0]);
    _mm_storeu_si128((__m128i *)dstGPtr, px128[1]);
    _mm_storeu_si128((__m128i *)dstBPtr, px128[2]);
}

inline void rpp_load24_f16pln3_to_f32pln3_avx(Rpp16f *srcPtrR, Rpp16f *srcPtrG, Rpp16f *srcPtrB, __m256 *p)
{
    p[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrR))));
    p[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrG))));
    p[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrB))));
}


inline void rpp_load48_i8pkd3_to_i8pln3(Rpp8s *srcPtr, __m128i *px)
{
    __m128i pxSrc[8];
    __m128i pxMask = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15);
    __m128i pxMaskRGB = _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);

    pxSrc[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    pxSrc[0] = _mm_shuffle_epi8(pxSrc[0], pxMask);    /* shuffle to get [R01|R02|R03|R04|G01|G02|G03|G04 || B01|B02|B03|B04|R05|G05|B05|R06] - Need R01-04, G01-04, B01-04 */
    pxSrc[1] = _mm_shuffle_epi8(pxSrc[1], pxMask);    /* shuffle to get [R05|R06|R07|R08|G05|G06|G07|G08 || B05|B06|B07|B08|R09|G09|B09|R10] - Need R05-08, G05-08, B05-08 */
    pxSrc[2] = _mm_shuffle_epi8(pxSrc[2], pxMask);    /* shuffle to get [R09|R10|R11|R12|G09|G10|G11|G12 || B09|B10|B11|B12|R13|G13|B13|R14] - Need R09-12, G09-12, B09-12 */
    pxSrc[3] = _mm_shuffle_epi8(pxSrc[3], pxMask);    /* shuffle to get [R13|R14|R15|R16|G13|G14|G15|G16 || B13|B14|B15|B16|R17|G17|B17|R18] - Need R13-16, G13-16, B13-16 */
    pxSrc[4] = _mm_unpacklo_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 lo-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[5] = _mm_unpacklo_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 lo-pixels of pxSrc[2] and pxSrc[3] */
    pxSrc[6] = _mm_unpackhi_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 hi-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[7] = _mm_unpackhi_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 hi-pixels of pxSrc[2] and pxSrc[3] */
    px[0] = _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB);    /* unpack 8 lo-pixels of pxSrc[4] and pxSrc[5] to get R01-16 */
    px[1] = _mm_shuffle_epi8(_mm_unpackhi_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB);    /* unpack 8 hi-pixels of pxSrc[4] and pxSrc[5] to get G01-16 */
    px[2] = _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[6], pxSrc[7]), pxMaskRGB);    /* unpack 8 lo-pixels of pxSrc[6] and pxSrc[7] to get B01-16 */
}

inline void rpp_load48_i8pkd3_to_u8pln3(Rpp8s *srcPtr, __m128i *px)
{
    __m128i pxSrc[8];
    __m128i pxMask = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15);
    __m128i pxMaskRGB = _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);

    pxSrc[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    pxSrc[0] = _mm_shuffle_epi8(pxSrc[0], pxMask);    /* shuffle to get [R01|R02|R03|R04|G01|G02|G03|G04 || B01|B02|B03|B04|R05|G05|B05|R06] - Need R01-04, G01-04, B01-04 */
    pxSrc[1] = _mm_shuffle_epi8(pxSrc[1], pxMask);    /* shuffle to get [R05|R06|R07|R08|G05|G06|G07|G08 || B05|B06|B07|B08|R09|G09|B09|R10] - Need R05-08, G05-08, B05-08 */
    pxSrc[2] = _mm_shuffle_epi8(pxSrc[2], pxMask);    /* shuffle to get [R09|R10|R11|R12|G09|G10|G11|G12 || B09|B10|B11|B12|R13|G13|B13|R14] - Need R09-12, G09-12, B09-12 */
    pxSrc[3] = _mm_shuffle_epi8(pxSrc[3], pxMask);    /* shuffle to get [R13|R14|R15|R16|G13|G14|G15|G16 || B13|B14|B15|B16|R17|G17|B17|R18] - Need R13-16, G13-16, B13-16 */
    pxSrc[4] = _mm_unpacklo_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 lo-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[5] = _mm_unpacklo_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 lo-pixels of pxSrc[2] and pxSrc[3] */
    pxSrc[6] = _mm_unpackhi_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 hi-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[7] = _mm_unpackhi_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 hi-pixels of pxSrc[2] and pxSrc[3] */
    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB));    /* unpack 8 lo-pixels of pxSrc[4] and pxSrc[5] to get R01-16 and add 128 to get u8 from i8 */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_shuffle_epi8(_mm_unpackhi_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB));    /* unpack 8 hi-pixels of pxSrc[4] and pxSrc[5] to get G01-16 and add 128 to get u8 from i8 */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[6], pxSrc[7]), pxMaskRGB));    /* unpack 8 lo-pixels of pxSrc[6] and pxSrc[7] to get B01-16 and add 128 to get u8 from i8 */
}

inline void rpp_store48_i8pln3_to_i8pln3(Rpp8s *dstPtrR, Rpp8s *dstPtrG, Rpp8s *dstPtrB, __m128i *px)
{
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_load48_i8pln3_to_i8pln3(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m128i *px)
{
    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_load48_i8pln3_to_u8pln3(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m128i *px)
{
    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrR));    /* load and convert to u8 [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrG));    /* load and convert to u8 [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrB));    /* load and convert to u8 [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_store48_i8pln3_to_i8pkd3(Rpp8s *dstPtr, __m128i *px)
{
    __m128i pxDst[4];
    __m128i pxZero = _mm_setzero_si128();
    __m128i pxMaskRGBAtoRGB = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
    pxDst[0] = _mm_unpacklo_epi8(px[1], pxZero);    /* unpack 8 lo-pixels of px[1] and pxZero */
    pxDst[1] = _mm_unpackhi_epi8(px[1], pxZero);    /* unpack 8 hi-pixels of px[1] and pxZero */
    pxDst[2] = _mm_unpacklo_epi8(px[0], px[2]);    /* unpack 8 lo-pixels of px[0] and px[2] */
    pxDst[3] = _mm_unpackhi_epi8(px[0], px[2]);    /* unpack 8 hi-pixels of px[0] and px[2] */
    _mm_storeu_si128((__m128i *)dstPtr, _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
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

RppStatus crop_f32_f32_host_standalone(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptROIPtr roiPtr,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    RpptROI roi;
    compute_roi_validation_host(roiPtr, &roi, &roiDefault, roiType);

    Rpp32f *srcPtrImage = srcPtr;
    Rpp32f *dstPtrImage = dstPtr;

    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    // Crop with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u alignedLength = (bufferLength / 12) * 12;

        Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        for(int i = 0; i < roi.xywhROI.roiHeight; i++)
        {
            Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow;
            dstPtrTempR = dstPtrRowR;
            dstPtrTempG = dstPtrRowG;
            dstPtrTempB = dstPtrRowB;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
            {
                __m128 p[4];
                rpp_load12_f32pkd3_to_f32pln3(srcPtrTemp, p);
                rpp_store12_f32pln3_to_f32pln3(dstPtrTempR, dstPtrTempG, dstPtrTempB, p);
                srcPtrTemp += 12;
                dstPtrTempR += 4;
                dstPtrTempG += 4;
                dstPtrTempB += 4;
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
        Rpp32u alignedLength = (bufferLength / 12) * 12;

        Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

        for(int i = 0; i < roi.xywhROI.roiHeight; i++)
        {
            Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR;
            srcPtrTempG = srcPtrRowG;
            srcPtrTempB = srcPtrRowB;
            dstPtrTemp = dstPtrRow;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
            {
                __m128 p[4];
                rpp_load12_f32pln3_to_f32pln3(srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                rpp_store12_f32pln3_to_f32pkd3(dstPtrTemp, p);
                srcPtrTempR += 4;
                srcPtrTempG += 4;
                srcPtrTempB += 4;
                dstPtrTemp += 12;
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
        Rpp32u copyLengthInBytes = bufferLength * sizeof(Rpp32f);

        for(int c = 0; c < layoutParams.channelParam; c++)
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    return RPP_SUCCESS;
}

RppStatus crop_f16_f16_host_standalone(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptROIPtr roiPtr,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    RpptROI roi;
    compute_roi_validation_host(roiPtr, &roi, &roiDefault, roiType);

    Rpp16f *srcPtrImage = srcPtr;
    Rpp16f *dstPtrImage = dstPtr;

    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

#if __AVX2__
    Rpp32u alignedLength = (bufferLength / 24) * 24;
    Rpp32u vectorIncrement = 24;
    Rpp32u vectorIncrementPerChannel = 8;
#else
    Rpp32u alignedLength = (bufferLength / 12) * 12;
    Rpp32u vectorIncrement = 12;
    Rpp32u vectorIncrementPerChannel = 4;
#endif

    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    // Crop with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        for(int i = 0; i < roi.xywhROI.roiHeight; i++)
        {
            Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow;
            dstPtrTempR = dstPtrRowR;
            dstPtrTempG = dstPtrRowG;
            dstPtrTempB = dstPtrRowB;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=vectorIncrement)
            {
#if __AVX2__
                __m256 p[3];
                rpp_load24_f16pkd3_to_f32pln3_avx(srcPtrTemp, p);    // simd loads
                rpp_store24_f32pln3_to_f16pln3_avx(dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];

                for(int cnt = 0; cnt < 12; cnt++)
                {
                    *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                }

                __m128 p[4];

                rpp_load12_f32pkd3_to_f32pln3(srcPtrTemp_ps, p);    // simd loads
                rpp_store12_f32pln3_to_f32pln3(dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                for(int cnt = 0; cnt < 4; cnt++)
                {
                    *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                    *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                }
#endif
                srcPtrTemp += vectorIncrement;
                dstPtrTempR += vectorIncrementPerChannel;
                dstPtrTempG += vectorIncrementPerChannel;
                dstPtrTempB += vectorIncrementPerChannel;
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
        Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

        for(int i = 0; i < roi.xywhROI.roiHeight; i++)
        {
            Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR;
            srcPtrTempG = srcPtrRowG;
            srcPtrTempB = srcPtrRowB;
            dstPtrTemp = dstPtrRow;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+= vectorIncrementPerChannel)
            {
#if __AVX2__
                __m256 p[3];
                rpp_load24_f16pln3_to_f32pln3_avx(srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                rpp_store24_f32pln3_to_f16pkd3_avx(dstPtrTemp, p);    // simd stores
#else
                Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                for(int cnt = 0; cnt < 4; cnt++)
                {
                    *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                    *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                    *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                }

                __m128 p[4];

                rpp_load12_f32pln3_to_f32pln3(srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                rpp_store12_f32pln3_to_f32pkd3(dstPtrTemp_ps, p);    // simd stores

                for(int cnt = 0; cnt < 12; cnt++)
                {
                    *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                }
#endif
                srcPtrTempR += vectorIncrementPerChannel;
                srcPtrTempG += vectorIncrementPerChannel;
                srcPtrTempB += vectorIncrementPerChannel;
                dstPtrTemp += vectorIncrement;
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
        Rpp32u copyLengthInBytes = bufferLength * sizeof(Rpp16f);

        for(int c = 0; c < layoutParams.channelParam; c++)
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            srcPtrChannel += srcDescPtr->strides.cStride;
            dstPtrChannel += dstDescPtr->strides.cStride;
        }
    }

    return RPP_SUCCESS;
}

RppStatus crop_i8_i8_host_standalone(Rpp8s *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8s *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiPtr,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    RpptROI roi;
    compute_roi_validation_host(roiPtr, &roi, &roiDefault, roiType);

    Rpp8s *srcPtrImage = srcPtr;
    Rpp8s *dstPtrImage = dstPtr;

    Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
    dstPtrChannel = dstPtrImage;

    // Crop with fused output-layout toggle (NHWC -> NCHW)
    if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u alignedLength = (bufferLength / 48) * 48;

        Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
        srcPtrRow = srcPtrChannel;
        dstPtrRowR = dstPtrChannel;
        dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
        dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

        for(int i = 0; i < roi.xywhROI.roiHeight; i++)
        {
            Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            srcPtrTemp = srcPtrRow;
            dstPtrTempR = dstPtrRowR;
            dstPtrTempG = dstPtrRowG;
            dstPtrTempB = dstPtrRowB;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
            {
                __m128i px[3];
                rpp_load48_i8pkd3_to_i8pln3(srcPtrTemp, px);    // simd loads
                rpp_store48_i8pln3_to_i8pln3(dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
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

        Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
        srcPtrRowR = srcPtrChannel;
        srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
        srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
        dstPtrRow = dstPtrChannel;

        for(int i = 0; i < roi.xywhROI.roiHeight; i++)
        {
            Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
            srcPtrTempR = srcPtrRowR;
            srcPtrTempG = srcPtrRowG;
            srcPtrTempB = srcPtrRowB;
            dstPtrTemp = dstPtrRow;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
            {
                __m128i px[3];
                rpp_load48_i8pln3_to_i8pln3(srcPtrTempR, srcPtrTempG, srcPtrTempB, px);    // simd loads
                rpp_store48_i8pln3_to_i8pkd3(dstPtrTemp, px);    // simd stores
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
            Rpp8s *srcPtrRow, *dstPtrRow;
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

struct PerformanceMetrics {
    double avg_time_ms, min_time_ms, max_time_ms, std_dev_ms;
    double fps, mpixels_per_sec, avg_latency_us;
    double data_read_mb, data_written_mb;
    double total_bandwidth_gbps, read_bandwidth_gbps, write_bandwidth_gbps;
    double pixels_processed;
    int bytes_per_pixel;
    double simd_utilization_percent;
    int vector_iterations, scalar_iterations;

    void print(const std::string& test_name) {
        std::cout << "\n┌─────────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│ " << std::left << std::setw(59) << test_name << " │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ TIMING METRICS                                              │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│   Average Time:           " << std::setw(10) << std::fixed << std::setprecision(3) 
                  << avg_time_ms << " ms                      │" << std::endl;
        std::cout << "│   Min Time:               " << std::setw(10) << min_time_ms << " ms                      │" << std::endl;
        std::cout << "│   Max Time:               " << std::setw(10) << max_time_ms << " ms                      │" << std::endl;
        std::cout << "│   Std Deviation:          " << std::setw(10) << std_dev_ms << " ms                      │" << std::endl;
        std::cout << "│   Latency (per frame):    " << std::setw(10) << std::setprecision(2) 
                  << avg_latency_us << " μs                     │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ THROUGHPUT METRICS                                          │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│   Frames Per Second:      " << std::setw(10) << fps << " FPS                     │" << std::endl;
        std::cout << "│   Megapixels/sec:         " << std::setw(10) << mpixels_per_sec << " MP/s                    │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ MEMORY BANDWIDTH METRICS                                    │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│   Data Read:              " << std::setw(10) << std::setprecision(3) 
                  << data_read_mb << " MB                      │" << std::endl;
        std::cout << "│   Data Written:           " << std::setw(10) << data_written_mb << " MB                      │" << std::endl;
        std::cout << "│   Total Bandwidth:        " << std::setw(10) << total_bandwidth_gbps << " GB/s                    │" << std::endl;
        std::cout << "│   Read Bandwidth:         " << std::setw(10) << read_bandwidth_gbps << " GB/s                    │" << std::endl;
        std::cout << "│   Write Bandwidth:        " << std::setw(10) << write_bandwidth_gbps << " GB/s                    │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ SIMD EFFICIENCY METRICS                                     │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│   SIMD Utilization:       " << std::setw(10) << std::setprecision(2) 
                  << simd_utilization_percent << " %                       │" << std::endl;
        std::cout << "│   Vector Iterations:      " << std::setw(10) << vector_iterations 
                  << "                            │" << std::endl;
        std::cout << "│   Scalar Iterations:      " << std::setw(10) << scalar_iterations 
                  << "                            │" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────────┘" << std::endl;
    }
};

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

PerformanceMetrics calculate_metrics(const std::vector<double>& times, int width, int height, 
                                    int bytes_per_pixel, bool has_layout_conversion) {
    PerformanceMetrics metrics = {};
    double sum = 0.0;
    metrics.min_time_ms = times[0];
    metrics.max_time_ms = times[0];
    
    for (double t : times) {
        sum += t;
        metrics.min_time_ms = std::min(metrics.min_time_ms, t);
        metrics.max_time_ms = std::max(metrics.max_time_ms, t);
    }
    
    metrics.avg_time_ms = sum / times.size();
    
    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - metrics.avg_time_ms) * (t - metrics.avg_time_ms);
    }
    metrics.std_dev_ms = std::sqrt(sq_sum / times.size());
    
    metrics.fps = 1000.0 / metrics.avg_time_ms;
    metrics.pixels_processed = width * height;
    metrics.mpixels_per_sec = (metrics.pixels_processed / 1e6) * metrics.fps;
    metrics.avg_latency_us = metrics.avg_time_ms * 1000.0;
    
    metrics.bytes_per_pixel = bytes_per_pixel;
    metrics.data_read_mb = (metrics.pixels_processed * bytes_per_pixel) / (1024.0 * 1024.0);
    metrics.data_written_mb = (metrics.pixels_processed * bytes_per_pixel) / (1024.0 * 1024.0);
    
    double total_data_mb = metrics.data_read_mb + metrics.data_written_mb;
    double time_seconds = metrics.avg_time_ms / 1000.0;
    
    metrics.total_bandwidth_gbps = (total_data_mb / 1024.0) / time_seconds;
    metrics.read_bandwidth_gbps = (metrics.data_read_mb / 1024.0) / time_seconds;
    metrics.write_bandwidth_gbps = (metrics.data_written_mb / 1024.0) / time_seconds;
    
    int pixels_per_row = width;
    if (has_layout_conversion) {
        metrics.vector_iterations = (pixels_per_row / 16) * height;
        metrics.scalar_iterations = (pixels_per_row % 16) * height;
    } else {
        metrics.vector_iterations = height;
        metrics.scalar_iterations = 0;
    }
    
    int total_iterations = metrics.vector_iterations + metrics.scalar_iterations;
    metrics.simd_utilization_percent = total_iterations > 0 ? 
        (100.0 * metrics.vector_iterations) / total_iterations : 0.0;
    
    return metrics;
}

void generate_test_image_nhwc(std::vector<Rpp8u>& img, int w, int h) {
    img.resize(w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            img[idx + 0] = (x * 255) / w;
            img[idx + 1] = (y * 255) / h;
            img[idx + 2] = 128;
        }
    }
}

void generate_test_image_nchw(std::vector<Rpp8u>& img, int w, int h) {
    img.resize(w * h * 3);
    int channelSize = w * h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            img[idx] = (x * 255) / w;
            img[idx + channelSize] = (y * 255) / h;
            img[idx + 2*channelSize] = 128;
        }
    }
}


void convert_nhwc_to_nchw(Rpp8u* src, Rpp8u* dst, int w, int h) {
    int channelSize = w * h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int nhwc_idx = (y * w + x) * 3;
            int nchw_idx = y * w + x;
            dst[nchw_idx] = src[nhwc_idx + 0];
            dst[nchw_idx + channelSize] = src[nhwc_idx + 1];
            dst[nchw_idx + 2*channelSize] = src[nhwc_idx + 2];
        }
    }
}

void convert_nchw_to_nhwc(Rpp8u* src, Rpp8u* dst, int w, int h) {
    int channelSize = w * h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int nchw_idx = y * w + x;
            int nhwc_idx = (y * w + x) * 3;
            dst[nhwc_idx + 0] = src[nchw_idx];
            dst[nhwc_idx + 1] = src[nchw_idx + channelSize];
            dst[nhwc_idx + 2] = src[nchw_idx + 2*channelSize];
        }
    }
}



struct CropConfig {
    std::string name;
    int x, y, width, height;
    std::string description;
};



int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "\nUsage: " << argv[0] << " <input_image>\n" << std::endl;
        return -1;
    }

    const char* input_path = argv[1];
    int srcW, srcH, srcC;

    unsigned char* pixels = stbi_load(input_path, &srcW, &srcH, &srcC, 3);
    if (!pixels) {
        std::cout << "Error: Could not load image " << input_path << std::endl;
        return -1;
    }

    std::cout << "\n===============================================================" << std::endl;
    std::cout << "  Comprehensive Crop Demo - NHWC AND NCHW Support!" << std::endl;
    std::cout << "===============================================================\n" << std::endl;
    std::cout << "Input: " << input_path << " (" << srcW << "x" << srcH << ")" << std::endl;

    // Store NHWC image
    std::vector<Rpp8u> srcImageNHWC(pixels, pixels + (srcW * srcH * 3));
    stbi_image_free(pixels); 

    // Convert to NCHW
    std::vector<Rpp8u> srcImageNCHW(srcW * srcH * 3);
    std::cout << "Converting to NCHW format..." << std::endl;
    convert_nhwc_to_nchw(srcImageNHWC.data(), srcImageNCHW.data(), srcW, srcH);
    std::cout << "Conversion complete\n" << std::endl;

    // Define crop size
    int cropW = std::min(300, srcW / 2);
    int cropH = std::min(300, srcH / 2);

    std::cout << "Crop size: " << cropW << "x" << cropH << " pixels\n" << std::endl;

    // Define all crop configurations
    std::vector<CropConfig> crops;

    // 1. CENTER CROP
    crops.push_back({"center", (srcW - cropW) / 2, (srcH - cropH) / 2, cropW, cropH, "Center"});

    // 2. CORNER CROPS
    crops.push_back({"top_left", 0, 0, cropW, cropH, "Top-left corner"});
    crops.push_back({"top_right", srcW - cropW, 0, cropW, cropH, "Top-right corner"});
    crops.push_back({"bottom_left", 0, srcH - cropH, cropW, cropH, "Bottom-left corner"});
    crops.push_back({"bottom_right", srcW - cropW, srcH - cropH, cropW, cropH, "Bottom-right corner"});

    // 3. EDGE MIDPOINT CROPS
    crops.push_back({"top_center", (srcW - cropW) / 2, 0, cropW, cropH, "Top edge center"});
    crops.push_back({"bottom_center", (srcW - cropW) / 2, srcH - cropH, cropW, cropH, "Bottom edge center"});
    crops.push_back({"left_center", 0, (srcH - cropH) / 2, cropW, cropH, "Left edge center"});
    crops.push_back({"right_center", srcW - cropW, (srcH - cropH) / 2, cropW, cropH, "Right edge center"});

    // 4. MARGIN CROPS (10% from edges)
    int margin = (int)(std::min(srcW, srcH) * 0.1);
    crops.push_back({"margin_top_left", margin, margin, cropW, cropH, "Top-left with margin"});
    crops.push_back({"margin_top_right", srcW - cropW - margin, margin, cropW, cropH, "Top-right with margin"});
    crops.push_back({"margin_bottom_left", margin, srcH - cropH - margin, cropW, cropH, "Bottom-left with margin"});
    crops.push_back({"margin_bottom_right", srcW - cropW - margin, srcH - cropH - margin, cropW, cropH, "Bottom-right with margin"});

    std::cout << "===============================================================" << std::endl;
    std::cout << "  Processing " << std::setw(2) << crops.size() << " crops in BOTH NHWC and NCHW formats" << std::endl;
    std::cout << "===============================================================\n" << std::endl;

    int successCountNHWC = 0;
    int successCountNCHW = 0;
    
    // Process each crop in BOTH formats
    for (size_t idx = 0; idx < crops.size(); idx++) {
        const auto& crop = crops[idx];
        
        std::cout << "---------------------------------------------------------------" << std::endl;
        std::cout << "  Crop " << std::setw(2) << (idx + 1) << "/" << std::setw(2) << crops.size() 
                  << ": " << std::left << std::setw(48) << crop.name << std::endl;
        std::cout << "  " << std::left << std::setw(59) << crop.description << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;

        int actualX = std::max(0, std::min(crop.x, srcW - 1));
        int actualY = std::max(0, std::min(crop.y, srcH - 1));
        int actualW = std::min(crop.width, srcW - actualX);
        int actualH = std::min(crop.height, srcH - actualY);

        if (actualW <= 0 || actualH <= 0) {
            std::cout << "  SKIPPED: Invalid dimensions\n" << std::endl;
            continue;
        }

        std::cout << "  Position: (" << actualX << ", " << actualY << ")" << std::endl;
        std::cout << "  Size: " << actualW << " x " << actualH << " pixels\n" << std::endl;

        RpptROI roi;
        roi.xywhROI.xy.x = actualX;
        roi.xywhROI.xy.y = actualY;
        roi.xywhROI.roiWidth = actualW;
        roi.xywhROI.roiHeight = actualH;

        std::cout << "  --- NHWC Processing -----------------------------------" << std::endl;
        {
            std::vector<Rpp8u> dstImage(actualW * actualH * 3);
            RpptDesc srcDesc = {0, 3, (Rpp32u)srcH, (Rpp32u)srcW, RpptLayout::NHWC, {0, (Rpp32u)srcW*3, 0, 1}};
            RpptDesc dstDesc = {0, 3, (Rpp32u)actualH, (Rpp32u)actualW, RpptLayout::NHWC, {0, (Rpp32u)actualW*3, 0, 1}};
            RppLayoutParams layoutParams = {3, 3};

            std::cout << "  Format: NHWC (Height x Width x Channels)" << std::endl;
            std::cout << "  Layout: Interleaved RGB [R G B][R G B]..." << std::endl;
            std::cout << "  Processing..." << std::endl;

            RppStatus status = crop_u8_u8_host_standalone(
                srcImageNHWC.data(), &srcDesc,
                dstImage.data(), &dstDesc,
                &roi, RpptRoiType::XYWH, layoutParams
            );

            if (status == RPP_SUCCESS) {
                std::string filename = "crop_nhwc_" + crop.name + ".jpg";
                if (stbi_write_jpg(filename.c_str(), actualW, actualH, 3, dstImage.data(), 90)) {
                    std::cout << "  SUCCESS: " << std::left << std::setw(37) << filename << std::endl;
                    successCountNHWC++;
                } else {
                    std::cout << "  FAILED to save file" << std::endl;
                }
            } else {
                std::cout << "  FAILED: Crop operation failed" << std::endl;
            }
        }
        std::cout << "  -------------------------------------------------------" << std::endl;

        std::cout << "  --- NCHW Processing -----------------------------------" << std::endl;
        {
            std::vector<Rpp8u> dstImageNCHW(actualW * actualH * 3);

            RpptDesc srcDesc = {0, 1, (Rpp32u)srcH, (Rpp32u)srcW, RpptLayout::NCHW, 
                               {0, (Rpp32u)srcW, 0, (Rpp32u)(srcW*srcH)}};
            RpptDesc dstDesc = {0, 1, (Rpp32u)actualH, (Rpp32u)actualW, RpptLayout::NCHW, 
                               {0, (Rpp32u)actualW, 0, (Rpp32u)(actualW*actualH)}};
            
            RppLayoutParams layoutParams = {1, 1};  

            std::cout << "  Format: NCHW (Channels x Height x Width)" << std::endl;
            std::cout << "  Layout: Planar [RR..][GG..][BB..]" << std::endl;
            std::cout << "  Processing 3 channels separately..." << std::endl;

            RppStatus status = RPP_SUCCESS;
            
            for (int c = 0; c < 3; c++) {
                Rpp8u* srcChannel = srcImageNCHW.data() + c * srcDesc.strides.cStride;
                Rpp8u* dstChannel = dstImageNCHW.data() + c * dstDesc.strides.cStride;
                
                std::cout << "    Channel " << c << " (" 
                          << (c == 0 ? "R" : c == 1 ? "G" : "B") << ")..." << std::endl;
                
                status = crop_u8_u8_host_standalone(
                    srcChannel, &srcDesc,
                    dstChannel, &dstDesc,
                    &roi, RpptRoiType::XYWH, layoutParams
                );
                
                if (status != RPP_SUCCESS) {
                    std::cout << "  FAILED: Channel " << c << " crop failed" << std::endl;
                    break;
                }
            }

            if (status == RPP_SUCCESS) {
                std::cout << "  Converting NCHW -> NHWC for saving..." << std::endl;
                std::vector<Rpp8u> dstImageForSave(actualW * actualH * 3);
                convert_nchw_to_nhwc(dstImageNCHW.data(), dstImageForSave.data(), actualW, actualH);
                
                std::string filename = "crop_nchw_" + crop.name + ".jpg";
                if (stbi_write_jpg(filename.c_str(), actualW, actualH, 3, dstImageForSave.data(), 90)) {
                    std::cout << "  SUCCESS: " << std::left << std::setw(37) << filename << std::endl;
                    successCountNCHW++;
                } else {
                    std::cout << "  FAILED to save file" << std::endl;
                }
            } else {
                std::cout << "  FAILED: Crop operation failed" << std::endl;
            }
        }
        std::cout << "  -------------------------------------------------------\n" << std::endl;
    
    }

    std::cout << "===============================================================" << std::endl;
    std::cout << "                    PROCESSING COMPLETE" << std::endl;
    std::cout << "===============================================================" << std::endl;
    std::cout << "  NHWC Crops: " << successCountNHWC << "/" << crops.size() << std::endl;
    std::cout << "  NCHW Crops: " << successCountNCHW << "/" << crops.size() << std::endl;
    std::cout << "  Total Files: " << (successCountNHWC + successCountNCHW) << std::endl;
    std::cout << "===============================================================\n" << std::endl;

    
    std::cout << "\n===============================================================" << std::endl;
    std::cout << "  Creating Format Comparison Visualization..." << std::endl;
    std::cout << "===============================================================\n" << std::endl;

    // Create a small sample image to show format difference
    const int sampleW = 4, sampleH = 3;
    std::vector<Rpp8u> sampleNHWC(sampleW * sampleH * 3);
    std::vector<Rpp8u> sampleNCHW(sampleW * sampleH * 3);

    // Fill with pattern: each pixel has unique RGB values
    for (int y = 0; y < sampleH; y++) {
        for (int x = 0; x < sampleW; x++) {
            int pixelNum = y * sampleW + x;
            Rpp8u r = (pixelNum * 30) % 256;
            Rpp8u g = (pixelNum * 50) % 256;
            Rpp8u b = (pixelNum * 70) % 256;
            
            // NHWC format: [R G B] [R G B] [R G B] ...
            int nhwc_idx = (y * sampleW + x) * 3;
            sampleNHWC[nhwc_idx + 0] = r;
            sampleNHWC[nhwc_idx + 1] = g;
            sampleNHWC[nhwc_idx + 2] = b;
        }
    }

    // Convert to NCHW
    convert_nhwc_to_nchw(sampleNHWC.data(), sampleNCHW.data(), sampleW, sampleH);

    // Print format comparison
    std::cout << "Format Comparison for " << sampleW << "×" << sampleH << " image:\n" << std::endl;
    
    std::cout << "NHWC Format (Height × Width × Channels):" << std::endl;
    std::cout << "Memory layout: Pixels stored with interleaved RGB" << std::endl;
    std::cout << "Pattern: [R0 G0 B0] [R1 G1 B1] [R2 G2 B2] ...\n" << std::endl;
    
    std::cout << "First 36 bytes (12 pixels) in NHWC:" << std::endl;
    for (int i = 0; i < std::min(36, (int)sampleNHWC.size()); i++) {
        if (i > 0 && i % 3 == 0) std::cout << " | ";
        std::cout << std::setw(3) << (int)sampleNHWC[i];
        if ((i + 1) % 3 == 0 && (i + 1) % 12 == 0) std::cout << "\n";
    }
    std::cout << std::endl;

    std::cout << "\nNCHW Format (Channels × Height × Width):" << std::endl;
    std::cout << "Memory layout: All R, then all G, then all B" << std::endl;
    std::cout << "Pattern: [R0 R1 R2 ...] [G0 G1 G2 ...] [B0 B1 B2 ...]\n" << std::endl;
    
    std::cout << "First 36 bytes in NCHW:" << std::endl;
    std::cout << "R channel: ";
    for (int i = 0; i < sampleW * sampleH; i++) {
        std::cout << std::setw(3) << (int)sampleNCHW[i] << " ";
    }
    std::cout << "\nG channel: ";
    for (int i = sampleW * sampleH; i < 2 * sampleW * sampleH; i++) {
        std::cout << std::setw(3) << (int)sampleNCHW[i] << " ";
    }
    std::cout << "\nB channel: ";
    for (int i = 2 * sampleW * sampleH; i < 3 * sampleW * sampleH; i++) {
        std::cout << std::setw(3) << (int)sampleNCHW[i] << " ";
    }
    std::cout << "\n" << std::endl;

    // Create a visual comparison image showing both formats side by side
    const int visW = 800, visH = 400;
    std::vector<Rpp8u> visualizationNHWC(visW * visH * 3, 255);
    
    // Draw format labels and data visualization
    // Left side: NHWC (interleaved)
    // Right side: NCHW (planar)
    
    // Create blocks showing memory layout
    int blockSize = 40;
    int startY = 100;
    
    // NHWC side (left)
    for (int i = 0; i < 12; i++) {
        int x = 50 + (i % 6) * (blockSize + 10);
        int y = startY + (i / 6) * (blockSize + 10);
        
        // Get RGB values for this pixel
        int pixelIdx = i / 3;
        int channel = i % 3;
        Rpp8u value = sampleNHWC[i];
        
        // Draw colored block
        for (int dy = 0; dy < blockSize; dy++) {
            for (int dx = 0; dx < blockSize; dx++) {
                int idx = ((y + dy) * visW + (x + dx)) * 3;
                if (idx + 2 < visualizationNHWC.size()) {
                    if (channel == 0) { // R
                        visualizationNHWC[idx + 0] = value;
                        visualizationNHWC[idx + 1] = 0;
                        visualizationNHWC[idx + 2] = 0;
                    } else if (channel == 1) { // G
                        visualizationNHWC[idx + 0] = 0;
                        visualizationNHWC[idx + 1] = value;
                        visualizationNHWC[idx + 2] = 0;
                    } else { // B
                        visualizationNHWC[idx + 0] = 0;
                        visualizationNHWC[idx + 1] = 0;
                        visualizationNHWC[idx + 2] = value;
                    }
                }
            }
        }
    }
    
    // NCHW side (right)
    for (int i = 0; i < 12; i++) {
        int x = 450 + (i % 4) * (blockSize + 10);
        int y = startY + (i / 4) * (blockSize + 10);
        
        Rpp8u value = sampleNCHW[i];
        int channel = i / 4;
        
        // Draw colored block
        for (int dy = 0; dy < blockSize; dy++) {
            for (int dx = 0; dx < blockSize; dx++) {
                int idx = ((y + dy) * visW + (x + dx)) * 3;
                if (idx + 2 < visualizationNHWC.size()) {
                    if (channel == 0) { // R channel
                        visualizationNHWC[idx + 0] = value;
                        visualizationNHWC[idx + 1] = 0;
                        visualizationNHWC[idx + 2] = 0;
                    } else if (channel == 1) { // G channel
                        visualizationNHWC[idx + 0] = 0;
                        visualizationNHWC[idx + 1] = value;
                        visualizationNHWC[idx + 2] = 0;
                    } else { // B channel
                        visualizationNHWC[idx + 0] = 0;
                        visualizationNHWC[idx + 1] = 0;
                        visualizationNHWC[idx + 2] = value;
                    }
                }
            }
        }
    }
    
    // Add title text area (simple colored bars)
    for (int y = 20; y < 60; y++) {
        for (int x = 50; x < 350; x++) {
            int idx = (y * visW + x) * 3;
            visualizationNHWC[idx + 0] = 100;
            visualizationNHWC[idx + 1] = 150;
            visualizationNHWC[idx + 2] = 255;
        }
        for (int x = 450; x < 750; x++) {
            int idx = (y * visW + x) * 3;
            visualizationNHWC[idx + 0] = 255;
            visualizationNHWC[idx + 1] = 150;
            visualizationNHWC[idx + 2] = 100;
        }
    }
    
    // Save visualization
    if (stbi_write_jpg("format_comparison.jpg", visW, visH, 3, visualizationNHWC.data(), 95)) {
        std::cout << "[OK] Created format_comparison.jpg - Visual comparison of NHWC vs NCHW" << std::endl;
    }

    // Create text file with detailed format information
    std::ofstream formatInfo("format_info.txt");
    if (formatInfo.is_open()) {
        
        formatInfo << "Example " << sampleW << "×" << sampleH << " image (" << (sampleW * sampleH) << " pixels):\n";
        formatInfo << "Memory Size: " << sampleNHWC.size() << " bytes\n";
        formatInfo << "Stride: " << (sampleW * 3) << " bytes per row\n\n";
        
        formatInfo << "First 36 bytes (12 pixels in NHWC):\n";
        for (int i = 0; i < std::min(36, (int)sampleNHWC.size()); i++) {
            if (i > 0 && i % 3 == 0) formatInfo << " | ";
            formatInfo << std::setw(3) << (int)sampleNHWC[i];
            if ((i + 1) % 3 == 0 && (i + 1) % 12 == 0) formatInfo << "\n";
        }
        formatInfo << "\n\n";
        
        
        formatInfo << "Example " << sampleW << "×" << sampleH << " image (" << (sampleW * sampleH) << " pixels):\n";
        formatInfo << "Memory Size: " << sampleNCHW.size() << " bytes\n";
        formatInfo << "Channel Stride: " << (sampleW * sampleH) << " bytes per channel\n";
        formatInfo << "Row Stride: " << sampleW << " bytes per row\n\n";
        
        formatInfo << "First 36 bytes in NCHW:\n";
        formatInfo << "R channel: ";
        for (int i = 0; i < sampleW * sampleH; i++) {
            formatInfo << std::setw(3) << (int)sampleNCHW[i] << " ";
        }
        formatInfo << "\nG channel: ";
        for (int i = sampleW * sampleH; i < 2 * sampleW * sampleH; i++) {
            formatInfo << std::setw(3) << (int)sampleNCHW[i] << " ";
        }
        formatInfo << "\nB channel: ";
        for (int i = 2 * sampleW * sampleH; i < 3 * sampleW * sampleH; i++) {
            formatInfo << std::setw(3) << (int)sampleNCHW[i] << " ";
        }
        formatInfo << "\n\n";
        
        
        formatInfo << "NHWC Format Files (" << successCountNHWC << " files):\n";
        for (const auto& crop : crops) {
            formatInfo << "  - crop_nhwc_" << crop.name << ".jpg\n";
        }
        
        formatInfo << "\nNCHW Format Files (" << successCountNCHW << " files):\n";
        for (const auto& crop : crops) {
            formatInfo << "  - crop_nchw_" << crop.name << ".jpg\n";
        }
        
        formatInfo << "\nTotal: " << (successCountNHWC + successCountNCHW) << " crop files\n";
        
        formatInfo.close();
        std::cout << "[OK] Created format_info.txt - Detailed format information" << std::endl;
    }

    std::cout << "\n===============================================================" << std::endl;
    std::cout << "  All files created successfully!" << std::endl;
    std::cout << "  - " << (successCountNHWC + successCountNCHW) << " crop images (NHWC + NCHW)" << std::endl;
    std::cout << "  - format_comparison.jpg (visual comparison)" << std::endl;
    std::cout << "  - format_info.txt (detailed information)" << std::endl;
    std::cout << "===============================================================\n" << std::endl;

    return 0;
}