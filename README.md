# Report COE 502 project
## Title: Parallelizing the Black Scholes Model
## Name: Abdulrhman Khormi
## ID: 201680920
## Date: 31/12/2023

# makefile commands:
compile the code:
```bash 
make all
```
run all the implementations:
```bash
make run-all
```

# 1. Introduction
The black scholes model is a mathematical model for pricing options contracts.
it is used to calculate the theoretical value of European-style options using current stock prices,
expected dividends, the option's strike price, expected interest rates,
time to expiration and expected volatility.
The model is widely used by options market participants to price options,
and it is also used by the exchanges themselves to set prices for newly introduced options contracts.
the model is also widely used by risk managers as a tool to monitor the reasonableness of prices
obtained by quoting options prices from financial models.
the model assumes the price of heavily traded assets follows a geometric Brownian motion
with constant drift and volatility.
when applied to a stock option, the model incorporates the constant price variation of the stock, 
the time value of money, the option's strike price and the time to the option's expiry.
I have implemented the black scholes model in C, and I have implemented three different implementations of the black scholes model.
the first implementation is a single threaded implementation of the black scholes model.
the second implementation is a vectorized implementation of the black scholes model.
the third implementation is a multithreading implementation of the black scholes model.

# 2. Baseline Implementation:
The baseline implementation is a simple implementation of the black scholes model.
it is a single threaded implementation that calculates the price of a call option.
CNDF is a helper function that calculates the cumulative normal distribution function.
this implementation is similar to the implementation provided in the project instructions.

```c
#define inv_sqrt_2xPI 0.39894228040143270286

float CNDF(float InputX) {
    int sign;

    float OutputX;
    float xInput;
    float xNPrimeofX;
    float expValues;
    float xK2;
    float xK2_2, xK2_3;
    float xK2_4, xK2_5;
    float xLocal, xLocal_1;
    float xLocal_2, xLocal_3;

    // Check for negative value of InputX
    if (InputX < 0.0) {
        InputX = -InputX;
        sign = 1;
    } else
        sign = 0;

    xInput = InputX;

    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-0.5f * InputX * InputX);
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = 0.2316419 * xInput;
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;

    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal = xLocal_1 * xNPrimeofX;
    xLocal = 1.0 - xLocal;

    OutputX = xLocal;

    if (sign) {
        OutputX = 1.0 - OutputX;
    }
    return OutputX;
}

float blackScholes(float sptprice, float strike, float rate, float volatility, float otime, int otype) {
    float OptionPrice;

    // local private working variables for the calculation
    float xStockPrice;
    float xStrikePrice;
    float xRiskFreeRate;
    float xVolatility;
    float xTime;
    float xSqrtTime;

    float logValues;
    float xLogTerm;
    float xD1;
    float xD2;
    float xPowerTerm;
    float xDen;
    float d1;
    float d2;
    float FutureValueX;
    float NofXd1;
    float NofXd2;
    float NegNofXd1;
    float NegNofXd2;

    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = otime;
    xSqrtTime = sqrt(xTime);

    logValues = log(sptprice / strike);
    xLogTerm = logValues;

    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;

    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;


    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 - xDen;

    d1 = xD1;
    d2 = xD2;

    NofXd1 = CNDF(d1);
    NofXd2 = CNDF(d2);

    float expval = exp(-rate * otime);
    FutureValueX = strike * expval;
    if (otype == 0) {
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else {
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }

    return OptionPrice;
}

/* Naive Implementation */
void *impl_scalar(void *args) {
    args_t *a = (args_t *) args;

    size_t i;
    for (i = 0; i < a->num_stocks; i++) {
        float sptPrice = a->sptPrice[i];
        float strike = a->strike[i];
        float rate = a->rate[i];
        float volatility = a->volatility[i];
        float otime = a->otime[i];
        char otype_c = a->otype[i];

        float otype = (tolower(otype_c) == 'p') ? 1 : 0;

        float value = blackScholes(sptPrice, strike, rate, volatility, otime, otype);

        a->output[i] = value;
    }
}
```

# 3. AVX2 Implementations:
AVX2 is an extension to the x86 instruction set architecture for microprocessors from Intel
and AMD proposed by Intel in March 2013. AVX2 exploits SIMD parallelism,
and it supports 256-bit floating point operations. Avx2 can perform eight floating point operations in parallel.
The AVX2 implementation is a vectorized implementation of the black scholes model.
It is a single threaded implementation of a black scholes model.
In this implementation, I tried to avoid if-else statements and loops.
I managed to avoid if-else by using the blendv instruction, which is a vectorized version of the ternary operator,
the cmp instruction, which is a vectorized version of the comparison operator,
and mask registers which are vectorized versions of the boolean variables.
I used log and exp functions from the vmath library provided in the project repository.
The reset of the code is similar to the scalar implementation just with vectorized instructions.
In CNDF function, I created a mask that contains ones in the positions where the input is negative,
and I created another mask that contains ones in the positions where the input is positive.
I used the mask to split the input into two vectors, one vector contains the positive inputs,
and the other vector contains the negative inputs.
I changed the sign of the negative inputs, and I added the two vectors together.
I used the mask again in output to select the negative inputs and subtract one from them.
The same idea is used with the output (option price) in the black scholes function.
```c
#define inv_sqrt_2xPI 0.39894228040143270286

__m256 mm256_mask_sub_ps(__m256 a, __m256 b, __m256 mask) {
    __m256 result = _mm256_sub_ps(a, b);
    return _mm256_blendv_ps(b, result, mask);
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

    NegNofXd1 = _mm256_sub_ps(_mm256_set1_ps(1.0f), NofXd1);
    NegNofXd2 = _mm256_sub_ps(_mm256_set1_ps(1.0f), NofXd2);

    __m256 stockxNofXd1 = _mm256_mul_ps(xStockPrice, NofXd1);
    __m256 FuturexNofXd2 = _mm256_mul_ps(FutureValueX, NofXd2);
    __m256 stockxNegNofXd1 = _mm256_mul_ps(xStockPrice, NegNofXd1);
    __m256 FuturexNegNofXd2 = _mm256_mul_ps(FutureValueX, NegNofXd2);

    OptionPrice = _mm256_sub_ps(stockxNofXd1, FuturexNofXd2);
    OptionPrice2 = _mm256_sub_ps(FuturexNegNofXd2, stockxNegNofXd1);

    OptionPrice = _mm256_blendv_ps(OptionPrice, OptionPrice2, otype);

    return OptionPrice;
}

/* Alternative Implementation */
void *impl_vector(void *args) {

    args_t *a = (args_t *) args;
    size_t num_stocks = a->num_stocks;

    __m256i omask = _mm256_set1_epi32(0x80000000);
    const int vlen = 32 / sizeof(float);

    for (size_t hw_vlen, i = 0; i < num_stocks; i += hw_vlen) {
        int rem = num_stocks - i;
        hw_vlen = rem < vlen ? rem : vlen;
        if (hw_vlen < vlen) {
            unsigned int m[vlen];
            for (size_t j = 0; j < vlen; j++)
                m[j] = (j < hw_vlen) ? 0x80000000 : 0x00000000;
            omask = _mm256_setr_epi32(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
        }
        __m256 sptprice_vec = _mm256_maskload_ps(a->sptPrice + i, omask);
        __m256 strike_vec = _mm256_maskload_ps(a->strike + i, omask);
        __m256 rate_vec = _mm256_maskload_ps(a->rate + i, omask);
        __m256 volatility_vec = _mm256_maskload_ps(a->volatility + i, omask);
        __m256 otime_vec = _mm256_maskload_ps(a->otime + i, omask);
        __m256i otype = _mm256_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (a->otype + i)));
        otype = _mm256_cmpeq_epi32(otype, _mm256_set1_epi32('P'));
        __m256 otype_vec = _mm256_castsi256_ps(otype);

        __m256 result = blackScholes_simd(sptprice_vec, strike_vec, rate_vec, volatility_vec, otime_vec, otype_vec);

        _mm256_maskstore_ps(a->output + i, omask, result);
    }

    return NULL;

}
```

# 4. pThreads Implementations:
pThreads is a POSIX standard for threads. It is a set of C programming language types, 
functions, and constants. The pThreads implementation is a multithreading implementation of the black scholes model.
It splits the dataset into chunks, and it processes each chunk in a separate thread.
Also, it computes the trailing elements in the main thread. The elements that are not enough to fill a chunk.
It extends the scalar implementation to use multiple threads:

```c
void *parallel(void *args) {
    args_t *a = (args_t *) args;
    for (size_t i = 0; i < a->num_stocks; i++) {
        float sptPrice = a->sptPrice[i];
        float strike = a->strike[i];
        float rate = a->rate[i];
        float volatility = a->volatility[i];
        float otime = a->otime[i];
        char otype_c = a->otype[i];

        float otype = (tolower(otype_c) == 'p') ? 1 : 0;

        a->output[i] = blackScholes_mimd(sptPrice, strike, rate, volatility, otime, otype);
    }
    return NULL;
}

/* Alternative Implementation */
void *impl_parallel(void *args) {

    args_t *a = (args_t *) args;
    /* Get all the arguments */
    register float *output = a->output;
    register float *sptPrice = a->sptPrice;
    register float *strike = a->strike;
    register float *rate = a->rate;
    register float *volatility = a->volatility;
    register float *otime = a->otime;
    register char *otype = a->otype;
    register size_t num_stocks = a->num_stocks;
    register size_t nthreads = a->nthreads;

    /* Create all threads */
    pthread_t tid[nthreads];
    args_t targs[nthreads];

    /* Assign current CPU to us */
    tid[0] = pthread_self();

    /* Amount of work per thread */
    size_t size_per_thread = num_stocks / nthreads;

    for (int i = 1; i < nthreads; i++) {
        /* Initialize the argument structure */
        targs[i].num_stocks = size_per_thread;
        targs[i].sptPrice = (sptPrice + (i * size_per_thread));
        targs[i].strike = (strike + (i * size_per_thread));
        targs[i].rate = (rate + (i * size_per_thread));
        targs[i].volatility = (volatility + (i * size_per_thread));
        targs[i].otime = (otime + (i * size_per_thread));
        targs[i].otype = (otype + (i * size_per_thread));
        targs[i].output = (output + (i * size_per_thread));
        pthread_create(&tid[i], NULL, parallel, &targs[i]);
    }

    /* Perform one portion of the work */
    for (size_t i = 0; i < size_per_thread; i++) {
        float sptPrice_i = sptPrice[i];
        float strike_i = strike[i];
        float rate_i = rate[i];
        float volatility_i = volatility[i];
        float otime_i = otime[i];
        char otype_i = otype[i];

        float otype = (tolower(otype_i) == 'p') ? 1 : 0;

        output[i] = blackScholes_mimd(sptPrice_i, strike_i, rate_i, volatility_i, otime_i, otype);
    }

    /* Perform trailing elements */
    int remaining = num_stocks % nthreads;
    for (size_t i = num_stocks - remaining; i < a->num_stocks; i++) {
        float sptPrice_i = sptPrice[i];
        float strike_i = strike[i];
        float rate_i = rate[i];
        float volatility_i = volatility[i];
        float otime_i = otime[i];
        char otype_i = otype[i];

        float otype = (tolower(otype_i) == 'p') ? 1 : 0;

        output[i] = blackScholes_mimd(sptPrice_i, strike_i, rate_i, volatility_i, otime_i, otype);
    }

    /* Join all threads */
    for (int i = 1; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }

    return NULL;
}
```
# 5. AVX2 & pThreads Implementations:
The AVX2 & pThreads implementation is a combination of the vector and parallel implementations.
It splits the dataset into chunks, and it processes each chunk in a separate thread.
Each thread will use the vector implementation to process its chunk.

```c
#include "vec.h"

void *impl_vector_para(void *args) {
    args_t *a = (args_t *) args;

    register float *output = a->output;
    register float *sptPrice = a->sptPrice;
    register float *strike = a->strike;
    register float *rate = a->rate;
    register float *volatility = a->volatility;
    register float *otime = a->otime;
    register char *otype = a->otype;
    register size_t num_stocks = a->num_stocks;
    register size_t nthreads = a->nthreads;

    pthread_t tid[nthreads];
    args_t targs[nthreads];

    tid[0] = pthread_self();

    size_t size_per_thread = num_stocks / nthreads;

    for (int i = 1; i < nthreads; i++) {
        targs[i].num_stocks = size_per_thread;
        targs[i].sptPrice = (sptPrice + (i * size_per_thread));
        targs[i].strike = (strike + (i * size_per_thread));
        targs[i].rate = (rate + (i * size_per_thread));
        targs[i].volatility = (volatility + (i * size_per_thread));
        targs[i].otime = (otime + (i * size_per_thread));
        targs[i].otype = (otype + (i * size_per_thread));
        targs[i].output = (output + (i * size_per_thread));
        pthread_create(&tid[i], NULL, impl_vector, &targs[i]);
    }

    for (size_t i = 0; i < size_per_thread; i += 8) {
        __m256 sptprice_vec = _mm256_loadu_ps(a->sptPrice + i);
        __m256 strike_vec = _mm256_loadu_ps(a->strike + i);
        __m256 rate_vec = _mm256_loadu_ps(a->rate + i);
        __m256 volatility_vec = _mm256_loadu_ps(a->volatility + i);
        __m256 otime_vec = _mm256_loadu_ps(a->otime + i);
        __m256i otype = _mm256_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (a->otype + i)));
        otype = _mm256_cmpeq_epi32(otype, _mm256_set1_epi32('P'));
        __m256 otype_vec = _mm256_castsi256_ps(otype);

        __m256 result = blackScholes_simd(sptprice_vec, strike_vec, rate_vec, volatility_vec, otime_vec, otype_vec);

        _mm256_storeu_ps(a->output + i, result);
    }

    int remaining = num_stocks % nthreads;
    __m256i omask = _mm256_set1_epi32(0x80000000);
    const int vlen = 32 / sizeof(float);

    for (size_t hw_vlen, i = num_stocks - remaining; i < num_stocks; i += hw_vlen) {
        int rem = num_stocks - i;
        hw_vlen = rem < vlen ? rem : vlen;
        if (hw_vlen < vlen) {
            unsigned int m[vlen];
            for (size_t j = 0; j < vlen; j++)
                m[j] = (j < hw_vlen) ? 0x80000000 : 0x00000000;
            omask = _mm256_setr_epi32(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
        }
        __m256 sptprice_vec = _mm256_maskload_ps(a->sptPrice + i, omask);
        __m256 strike_vec = _mm256_maskload_ps(a->strike + i, omask);
        __m256 rate_vec = _mm256_maskload_ps(a->rate + i, omask);
        __m256 volatility_vec = _mm256_maskload_ps(a->volatility + i, omask);
        __m256 otime_vec = _mm256_maskload_ps(a->otime + i, omask);
        __m256i otype = _mm256_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (a->otype + i)));
        otype = _mm256_cmpeq_epi32(otype, _mm256_set1_epi32('P'));
        __m256 otype_vec = _mm256_castsi256_ps(otype);

        __m256 result = blackScholes_simd(sptprice_vec, strike_vec, rate_vec, volatility_vec, otime_vec, otype_vec);

        _mm256_maskstore_ps(a->output + i, omask, result);
    }

    for (int i = 1; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }

    return NULL;

}
```

# 6. Results:
### The following table shows the results of the different implementations on different datasets on two different machines.

## machine one (11th Gen Intel(R) Core(TM) i7-1165G7 8 threads):
| Implementation | Time (ns) | Speedup | dataset |
|----------------|-----------|---------|---------|
| scalar         | 3192      | 1.0000  | dev     |
| vector         | 641       | 4.9797  | dev     |
| parallel       | 76819     | 0.0416  | dev     |
| scalar         | 180545    | 1.0000  | small   |
| vector         | 32330     | 5.5844  | small   |
| parallel       | 105888    | 1.7050  | small   |
| scalar         | 708408    | 1.0000  | medium  |
| vector         | 128486    | 5.5135  | medium  |
| parallel       | 206416    | 3.4319  | medium  |
| scalar         | 2838532   | 1.0000  | large   |
| vector         | 518506    | 5.4744  | large   |
| parallel       | 647415    | 4.3844  | large   |
| scalar         | 458689072 | 1.0000  | native  |
| vector         | 83608093  | 5.4861  | native  |
| parallel       | 116701107 | 3.9304  | native  |

## machine two (AMD Ryzen 9 5900X 24 threads):
| Implementation | Time (ns) | Speedup | dataset |
|----------------|-----------|---------|---------|
| scalar         | 891       | 1.0000  | dev     |
| vector         | 149       | 5.9798  | dev     |
| parallel       | 515554    | 0.0017  | dev     |
| scalar         | 121671    | 1.0000  | small   |
| vector         | 23412     | 5.1969  | small   |
| parallel       | 500812    | 0.2429  | small   |
| scalar         | 483029    | 1.0000  | medium  |
| vector         | 94060     | 5.1353  | medium  |
| parallel       | 522427    | 0.9245  | medium  |
| scalar         | 1933590   | 1.0000  | large   |
| vector         | 376048    | 5.1418  | large   |
| parallel       | 591913    | 3.2666  | large   |
| scalar         | 304103865 | 1.0000  | native  |
| vector         | 58929702  | 5.1604  | native  |
| parallel       | 21180817  | 14.3575 | native  |

formula for speedup: speedup = time of scalar implementation / time of implementation

As we can see from the results, the vector implementation shows impressive performance improvement with 5x speedup.
It almost outperforms the parallel implementation in all datasets.
But the parallel implementation is faster in larger datasets.
The parallel implementation speedup in 24 threads is 14x, but the speedup in eight threads is only 4x.
Parallel implementation didn't perform well in small datasets, but it performed well in large datasets.
That is because the overhead of creating threads is very high,
and it is not worth it in small datasets.
Also, adding more threads will not always improve the performance,
but it will increase the overhead of creating threads thus decreasing the performance.
The overhead of creating threads is tiny in large datasets, and it is worth it in large datasets.
The vector implementation performed well in small datasets,
but it didn't perform as good as parallel implementation in native dataset.
That is because the vector implementation is limited by the size of the vector registers.
The vector registers are 256 bits, and they can hold 8 floats.


# 7. Conclusion:
Both the vector and parallel implementations show a significant speedup over the scalar implementation.
The parallel implementation is easier to implement than the vector implementation.
The parallel didn't perform well in small datasets, but it performed well in large datasets.
That is because the overhead of creating threads is very high, and it is not worth it in small datasets.
The overhead of creating threads is tiny in large datasets, and it is worth it in large datasets. 
The vector implementation performed well in small datasets, but it didn't perform as good as parallel 
implementation in large datasets. That is because the vector implementation is limited by the size 
of the vector registers. The vector registers are 256 bits, and they can hold 8 floats. 
We can improve the vector implementation by using the AVX512 registers.
The AVX512 registers are 512 bits, and they can hold 16 floats.
But the AVX512 registers are not supported by all processors.
Also, we can combine the vector and parallel implementations to get the best of both worlds. 
We can divide the dataset into chunks, and we can process each chunk in a separate thread.
Each thread will use the vector implementation to process its chunk.
This will reduce the overhead of creating threads,
and it will improve the performance of the vector implementation. 
I think the vector implementation is the best implementation for this problem because it shows consistent performance
improvement unlike the parallel implementation.
But by combining the vector and parallel implementations, The resulting implementation will have the best performance
in native dataset and large datasets. Combining the vector and parallel got 19x speedup in native dataset in machine one,
and it got 44x speedup in native dataset in machine two. The time difference between the baseline implementation and the
combined implementation is extremely noticeable.

# 8. References
https://github.com/hawajkm/characterize-microbenchmark.git \
https://github.com/AbdulrhmanKhormi/coe502.git

# 9. Machine and OS Specifications

# machine one
## cpu info:
```
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         39 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  8
  On-line CPU(s) list:   0-7
Vendor ID:               GenuineIntel
  Model name:            11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
    CPU family:          6
    Model:               140
    Thread(s) per core:  2
    Core(s) per socket:  4
    Socket(s):           1
    Stepping:            1
    CPU(s) scaling MHz:  25%
    CPU max MHz:         4700.0000
    CPU min MHz:         400.0000
    BogoMIPS:            5606.40
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl x
                         topology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx 
                         f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l2 cdp_l2 ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdt_a a
                         vx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves split_lock_detect user_shstk dtherm ida arat pln pts hwp hwp_notify hw
                         p_act_window hwp_epp hwp_pkg_req vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid movdiri movdir64b fsrm avx512_vp2intersect md_clear ibt 
                         flush_l1d arch_capabilities
Virtualization features: 
  Virtualization:        VT-x
Caches (sum of all):     
  L1d:                   192 KiB (4 instances)
  L1i:                   128 KiB (4 instances)
  L2:                    5 MiB (4 instances)
  L3:                    12 MiB (1 instance)
NUMA:                    
  NUMA node(s):          1
  NUMA node0 CPU(s):     0-7
Vulnerabilities:         
  Gather data sampling:  Mitigation; Microcode
  Itlb multihit:         Not affected
  L1tf:                  Not affected
  Mds:                   Not affected
  Meltdown:              Not affected
  Mmio stale data:       Not affected
  Retbleed:              Not affected
  Spec rstack overflow:  Not affected
  Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:            Mitigation; Enhanced / Automatic IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
  Srbds:                 Not affected
  Tsx async abort:       Not affected
```

## OS info:
```
Linux fedora 6.6.8-200.fc39.x86_64 #1 SMP PREEMPT_DYNAMIC Thu Dec 21 04:01:49 UTC 2023 x86_64 GNU/Linux
```

## gcc info:
```
gcc (GCC) 13.2.1 20231205 (Red Hat 13.2.1-6)
Copyright (C) 2023 Free Software Foundation, Inc.
```

## machine two
## cpu info:
```
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         48 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  24
  On-line CPU(s) list:   0-23
Vendor ID:               AuthenticAMD
  Model name:            AMD Ryzen 9 5900X 12-Core Processor
    CPU family:          25
    Model:               33
    Thread(s) per core:  2
    Core(s) per socket:  12
    Socket(s):           1
    Stepping:            2
    Frequency boost:     enabled
    CPU(s) scaling MHz:  52%
    CPU max MHz:         4950.1948
    CPU min MHz:         2200.0000
    BogoMIPS:            7400.17
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe
                          popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2
                          erms invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk clzero irperf xsaveerptr rdpru wbnoinvd arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefi
                         lter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smca fsrm debug_swap
Virtualization features: 
  Virtualization:        AMD-V
Caches (sum of all):     
  L1d:                   384 KiB (12 instances)
  L1i:                   384 KiB (12 instances)
  L2:                    6 MiB (12 instances)
  L3:                    64 MiB (2 instances)
NUMA:                    
  NUMA node(s):          1
  NUMA node0 CPU(s):     0-23
Vulnerabilities:         
  Gather data sampling:  Not affected
  Itlb multihit:         Not affected
  L1tf:                  Not affected
  Mds:                   Not affected
  Meltdown:              Not affected
  Mmio stale data:       Not affected
  Retbleed:              Not affected
  Spec rstack overflow:  Vulnerable: Safe RET, no microcode
  Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:            Mitigation; Retpolines, IBPB conditional, IBRS_FW, STIBP always-on, RSB filling, PBRSB-eIBRS Not affected
  Srbds:                 Not affected
  Tsx async abort:       Not affected
```

## OS info:
```
Linux fedora 6.6.8-200.fc39.x86_64 #1 SMP PREEMPT_DYNAMIC Thu Dec 21 04:01:49 UTC 2023 x86_64 GNU/Linux
```

## gcc info:
```
gcc (GCC) 13.2.1 20231205 (Red Hat 13.2.1-6)
Copyright (C) 2023 Free Software Foundation, Inc.
```