/* vec.c
 *
 * Author:
 * Date  :
 *
 *  Description
 */

/* Standard C includes */
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <ctype.h>
#include <pthread.h>

/* Include common headers */
#include "../../common/types.h"
#include "../../common/macros.h"

/* Include application-specific headers */
#include "../include/types.h"
#include "../../common/vmath.h"

#define inv_sqrt_2xPI 0.39894228040143270286

__m256 mm256_mask_sub_ps(__m256 a, __m256 b, __m256 mask) {
    __m256 result = _mm256_sub_ps(a, b);
    return _mm256_blendv_ps(b, result, mask);
}

__m256 mm256_maskz_sub_ps(__m256 a, __m256 b, __m256 mask) {
    __m256 result = _mm256_sub_ps(a, b);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, mask);
}

__m256 mm256_maskz_loadu_ps(__m256 b, __m256 mask) {
    return _mm256_and_ps(b, mask);
}

__m256 mm256_LT_mask(__m256 a, __m256 b) {
    __m256 result = _mm256_cmp_ps(a, b, _CMP_LT_OQ);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, result);
}

__m256 mm256_GE_mask(__m256 a, __m256 b) {
    __m256 result = _mm256_cmp_ps(a, b, _CMP_GE_OQ);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, result);
}

__m256 mm256_EQ_mask(__m256 a, __m256 b) {
    __m256 result = _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, result);
}

__m256 mm256_NEQ_mask(__m256 a, __m256 b) {
    __m256 result = _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, result);
}

__m256 CNDF_simd(__m256 InputX) {

    __m256 OutputX;
    __m256 xInput;
    __m256 xInputNegative;
    __m256 xNPrimeofX;
    __m256 expValues;
    __m256 xK2;
    __m256 xK2_2, xK2_3;
    __m256 xK2_4, xK2_5;
    __m256 xLocal, xLocal_1;
    __m256 xLocal_2, xLocal_3;

    // spilt the input vector into positive and negative vectors
    __m256 mask = mm256_LT_mask(_mm256_setzero_ps(), InputX);
    __m256 mask2 = mm256_GE_mask(_mm256_setzero_ps(), InputX);
    xInput = mm256_maskz_loadu_ps(InputX, mask);
    xInputNegative = mm256_maskz_loadu_ps(InputX, mask2);

    // make negative value of InputNegative to positive
    xInputNegative = _mm256_mul_ps(xInputNegative, _mm256_set1_ps(-1.0f));

    // add positive and negative values
    xInput = _mm256_add_ps(xInput, xInputNegative);

    expValues = _mm256_exp_ps(_mm256_mul_ps(_mm256_set1_ps(-0.5f), _mm256_mul_ps(InputX, InputX)));

    xNPrimeofX = expValues;
    xNPrimeofX = _mm256_mul_ps(xNPrimeofX, _mm256_set1_ps(inv_sqrt_2xPI));

    xK2 = _mm256_mul_ps(_mm256_set1_ps(0.2316419), xInput);
    xK2 = _mm256_add_ps(_mm256_set1_ps(1.0f), xK2);
    xK2 = _mm256_div_ps(_mm256_set1_ps(1.0f), xK2);
    xK2_2 = _mm256_mul_ps(xK2, xK2);
    xK2_3 = _mm256_mul_ps(xK2_2, xK2);
    xK2_4 = _mm256_mul_ps(xK2_3, xK2);
    xK2_5 = _mm256_mul_ps(xK2_4, xK2);

    xLocal_1 = _mm256_mul_ps(xK2, _mm256_set1_ps(0.319381530));
    xLocal_2 = _mm256_mul_ps(xK2_2, _mm256_set1_ps(-0.356563782));
    xLocal_3 = _mm256_mul_ps(xK2_3, _mm256_set1_ps(1.781477937));
    xLocal_2 = _mm256_add_ps(xLocal_2, xLocal_3);
    xLocal_3 = _mm256_mul_ps(xK2_4, _mm256_set1_ps(-1.821255978));
    xLocal_2 = _mm256_add_ps(xLocal_2, xLocal_3);
    xLocal_3 = _mm256_mul_ps(xK2_5, _mm256_set1_ps(1.330274429));
    xLocal_2 = _mm256_add_ps(xLocal_2, xLocal_3);

    xLocal_1 = _mm256_add_ps(xLocal_2, xLocal_1);
    xLocal = _mm256_mul_ps(xLocal_1, xNPrimeofX);
    xLocal = _mm256_sub_ps(_mm256_set1_ps(1.0f), xLocal);

    OutputX = xLocal;
    OutputX = mm256_mask_sub_ps(_mm256_set1_ps(1.0f), OutputX, mask2);

    return OutputX;
}

__m256 blackScholes_simd(__m256 sptprice, __m256 strike, __m256 rate, __m256 volatility, __m256 otime, __m256 otype) {
    __m256 OptionPrice;
    __m256 OptionPrice2;

    // local private working variables for the calculation
    __m256 xStockPrice;
    __m256 xStrikePrice;
    __m256 xRiskFreeRate;
    __m256 xVolatility;
    __m256 xTime;
    __m256 xSqrtTime;

    __m256 xLogTerm;
    __m256 xD1;
    __m256 xD2;
    __m256 xPowerTerm;
    __m256 xDen;
    __m256 d1;
    __m256 d2;
    __m256 FutureValueX;
    __m256 NofXd1;
    __m256 NofXd2;
    __m256 NegNofXd1;
    __m256 NegNofXd2;

    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = otime;
    xSqrtTime = _mm256_sqrt_ps(xTime);

    xLogTerm = _mm256_log_ps(_mm256_div_ps(xStockPrice, xStrikePrice));
    xPowerTerm = _mm256_mul_ps(xVolatility, xVolatility);
    xPowerTerm = _mm256_mul_ps(xPowerTerm, _mm256_set1_ps(0.5));

    xD1 = _mm256_add_ps(xRiskFreeRate, xPowerTerm);
    xD1 = _mm256_mul_ps(xD1, xTime);
    xD1 = _mm256_add_ps(xD1, xLogTerm);

    xDen = _mm256_mul_ps(xVolatility, xSqrtTime);
    xD1 = _mm256_div_ps(xD1, xDen);
    xD2 = _mm256_sub_ps(xD1, xDen);

    d1 = xD1;
    d2 = xD2;

    NofXd1 = CNDF_simd(d1);
    NofXd2 = CNDF_simd(d2);

    __m256 expVal = _mm256_exp_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), _mm256_mul_ps(rate, otime)));

    FutureValueX = _mm256_mul_ps(xStrikePrice, expVal);

    __m256 mask = mm256_EQ_mask(otype, _mm256_setzero_ps());
    __m256 mask2 = mm256_NEQ_mask(otype, _mm256_setzero_ps());

    NegNofXd1 = mm256_maskz_sub_ps(_mm256_set1_ps(1.0f), NofXd1, mask2);
    NegNofXd2 = mm256_maskz_sub_ps(_mm256_set1_ps(1.0f), NofXd2, mask2);

    __m256 stockxNofXd1 = _mm256_mul_ps(xStockPrice, NofXd1);
    __m256 FuturexNofXd2 = _mm256_mul_ps(FutureValueX, NofXd2);
    __m256 stockxNegNofXd1 = _mm256_mul_ps(xStockPrice, NegNofXd1);
    __m256 FuturexNegNofXd2 = _mm256_mul_ps(FutureValueX, NegNofXd2);

    OptionPrice = mm256_maskz_sub_ps(stockxNofXd1, FuturexNofXd2, mask);
    OptionPrice2 = mm256_maskz_sub_ps(FuturexNegNofXd2, stockxNegNofXd1, mask2);

    OptionPrice = _mm256_add_ps(OptionPrice, OptionPrice2);

    return OptionPrice;
}

__m256 char_to_float(char *c) {
    __m256 result;
    for (int i = 0; i < 8; ++i) {
        result[i] = (tolower(c[i]) == 'p') ? 1 : 0;
    }
    return result;
}

/* Alternative Implementation */
void *impl_vector(void *args) {

    args_t *a = (args_t *) args;

    size_t i;
    size_t num_stocks = a->num_stocks;

    for (i = 0; i < num_stocks; i += 8) {
        __m256 sptprice_vec = _mm256_loadu_ps(a->sptPrice + i);
        __m256 strike_vec = _mm256_loadu_ps(a->strike + i);
        __m256 rate_vec = _mm256_loadu_ps(a->rate + i);
        __m256 volatility_vec = _mm256_loadu_ps(a->volatility + i);
        __m256 otime_vec = _mm256_loadu_ps(a->otime + i);
        __m256 otype_vec = char_to_float(a->otype + i);

        __m256 result = blackScholes_simd(sptprice_vec, strike_vec, rate_vec, volatility_vec, otime_vec, otype_vec);

        _mm256_storeu_ps(a->output + i, result);
    }

    return NULL;

}