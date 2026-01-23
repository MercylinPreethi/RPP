/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// #include "host_tensor_executors.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <immintrin.h>  // Includes all SIMD headers (SSE/AVX)

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- RPP Type Definitions ---
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
struct RpptROI { RpptXYWH xywhROI; };
typedef RpptROI* RpptROIPtr;

struct RpptStride { Rpp32u nStride; Rpp32u hStride; Rpp32u wStride; Rpp32u cStride; };
struct RpptDesc { Rpp32u n; Rpp32u c; Rpp32u h; Rpp32u w; RpptLayout layout; RpptStride strides; };
typedef RpptDesc* RpptDescPtr;

struct RppLayoutParams { Rpp32u bufferMultiplier; Rpp32u channelParam; };

// Global SIMD constants
static const __m128i xmm_pxConvertI8 = _mm_set1_epi8((char)128);

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

inline void rpp_load48_u8pkd3_to_u8pln3(Rpp8u *srcPtr, __m128i *px)
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

inline void rpp_store48_u8pln3_to_u8pln3(Rpp8u *dstPtrR, Rpp8u *dstPtrG, Rpp8u *dstPtrB, __m128i *px)
{
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_load48_u8pln3_to_u8pln3(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m128i *px)
{
    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
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
    _mm_storeu_si128((__m128i *)dstPtr, _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
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

inline void rpp_load24_f16pln3_to_f32pln3_avx(Rpp16f *srcPtrR, Rpp16f *srcPtrG, Rpp16f *srcPtrB, __m256 *p)
{
    p[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrR))));
    p[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrG))));
    p[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrB))));
}

inline void rpp_load48_f16pln3_to_f32pln3_avx(Rpp16f *srcPtrR, Rpp16f *srcPtrG, Rpp16f *srcPtrB, __m256 *p)
{
    p[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrR))));
    p[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrR + 8))));
    p[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrG))));
    p[3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrG + 8))));
    p[4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrB))));
    p[5] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrB + 8))));
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
                rpp_load48_u8pkd3_to_u8pln3(srcPtrTemp, px);    // simd loads
                rpp_store48_u8pln3_to_u8pln3(dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
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
                rpp_load48_u8pln3_to_u8pln3(srcPtrTempR, srcPtrTempG, srcPtrTempB, px);    // simd loads
                rpp_store48_u8pln3_to_u8pkd3(dstPtrTemp, px);    // simd stores
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
                rpp_load12_f32pkd3_to_f32pln3(srcPtrTemp, p);    // simd loads
                rpp_store12_f32pln3_to_f32pln3(dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
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
                rpp_load12_f32pln3_to_f32pln3(srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                rpp_store12_f32pln3_to_f32pkd3(dstPtrTemp, p);    // simd stores
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
                rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];

                for(int cnt = 0; cnt < 12; cnt++)
                {
                    *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                }

                __m128 p[4];

                rpp_load12_f32pkd3_to_f32pln3(srcPtrTemp_ps, p);    // simd loads
                rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

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