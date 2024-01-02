/* vec.h
 *
 * Author: Khalid Al-Hawaj
 * Date  : 13 Nov. 2023
 *
 * Header for the vectorized function.
 */
#include "immintrin.h"

#ifndef __IMPL_VEC_H_
#define __IMPL_VEC_H_

/* Function declaration */
void *impl_vector(void *args);

__m256 char_to_float(char *c);

__m256 blackScholes_simd(__m256 sptprice, __m256 strike, __m256 rate, __m256 volatility, __m256 otime, __m256 otype);

#endif //__IMPL_VEC_H_
