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

float CNDF(float InputX){
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
    xLocal   = xLocal_1 * xNPrimeofX;
    xLocal   = 1.0 - xLocal;

    OutputX  = xLocal;

    if (sign) {
        OutputX = 1.0 - OutputX;
    }
    return OutputX;
}

float blackScholes(float sptprice, float strike, float rate, float volatility, float otime, int otype){
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
void* impl_scalar(void* args)
{
    args_t* a = (args_t*) args;

    size_t i;
    for (i = 0; i < a->num_stocks; i++) {
        float sptPrice   = a->sptPrice  [i];
        float strike     = a->strike    [i];
        float rate       = a->rate      [i];
        float volatility = a->volatility[i];
        float otime      = a->otime     [i];
        char  otype_c     = a->otype     [i];

        float otype = (tolower(otype_c) == 'p')? 1 : 0;

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
Unfortunately, I couldn't avoid the loop for log and exp functions.
The reset of the code is similar to the scalar implementation just with vectorized instructions.
In CNDF function, I created a mask that contains ones in the positions where the input is negative,
and I created another mask that contains ones in the positions where the input is positive.
I used the mask to split the input into two vectors, one vector contains the positive inputs,
and the other vector contains the negative inputs.
I changed the sign of the negative inputs, and I added the two vectors together.
I used the mask again in output to select the negative inputs and subtract one from them.
The same idea is used in the black scholes function.
```c
#define inv_sqrt_2xPI 0.39894228040143270286

__m256 mm256_mask_sub_ps(__m256 a, __m256 b, __m256 mask){
    __m256 result = _mm256_sub_ps(a, b);
    return _mm256_blendv_ps(b, result, mask);
}

__m256 mm256_maskz_sub_ps(__m256 a, __m256 b, __m256 mask){
    __m256 result = _mm256_sub_ps(a, b);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, mask);
}

__m256 mm256_maskz_loadu_ps(__m256 b, __m256 mask){
    return _mm256_and_ps(b, mask);
}

__m256 mm256_LT_mask(__m256 a, __m256 b){
    __m256 result = _mm256_cmp_ps(a, b, _CMP_LT_OQ);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, result);
}

__m256 mm256_GE_mask(__m256 a, __m256 b){
    __m256 result = _mm256_cmp_ps(a, b, _CMP_GE_OQ);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, result);
}

__m256 mm256_EQ_mask(__m256 a, __m256 b){
    __m256 result = _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, result);
}

__m256 mm256_NEQ_mask(__m256 a, __m256 b){
    __m256 result = _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
    return _mm256_blendv_ps(_mm256_setzero_ps(), result, result);
}

__m256 CNDF_simd(__m256 InputX){
    __mmask8 sign;

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
    
    __m256 mask = mm256_LT_mask(_mm256_setzero_ps(), InputX);
    __m256 mask2 = mm256_GE_mask(_mm256_setzero_ps(), InputX);
    xInput = mm256_maskz_loadu_ps(InputX, mask);
    xInputNegative = mm256_maskz_loadu_ps(InputX, mask2);
    
    xInputNegative = _mm256_mul_ps(xInputNegative, _mm256_set1_ps(-1.0f));

    xInput = _mm256_add_ps(xInput, xInputNegative);

    for (int i = 0; i < 8; ++i) {
        expValues[i] = exp(-0.5f * InputX[i] * InputX[i]);
    }

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
    xLocal   = _mm256_mul_ps(xLocal_1, xNPrimeofX);
    xLocal   = _mm256_sub_ps(_mm256_set1_ps(1.0f), xLocal);

    OutputX  = xLocal;
    OutputX = mm256_mask_sub_ps( _mm256_set1_ps(1.0f),OutputX ,mask2 );

    return OutputX;
}

__m256 blackScholes_simd(__m256 sptprice, __m256 strike, __m256 rate, __m256 volatility, __m256 otime,__m256 otype){
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

    float logValues[8];

    for(int i=0; i<8;i ++) {
        logValues[i] = log(sptprice[i] / strike[i]);
    }

    xLogTerm = _mm256_loadu_ps(logValues);
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

    __m256 expVal;
    for (int i = 0; i < 8; ++i) {
        expVal[i] = exp(-(rate[i])*(otime[i]));
    }

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

__m256 char_to_float(char* c) {
    __m256 result;
    for (int i = 0; i < 8; ++i) {
        result[i] = (tolower(c[i]) == 'p')? 1 : 0;
    }
    return result;
}

/* Alternative Implementation */
void* impl_vector(void* args)
{

    args_t* a = (args_t*) args;

    size_t i;
    size_t num_stocks = a->num_stocks;

    float* sptprice   = a->sptPrice  ;
    float* strike     = a->strike    ;
    float* rate       = a->rate      ;
    float* volatility = a->volatility;
    float* otime      = a->otime     ;
    char * otype      = a->otype     ;
    float* output     = a->output    ;

    for (i = 0; i < num_stocks; i+=8) {
        __m256 sptprice_vec = _mm256_loadu_ps(sptprice + i);
        __m256 strike_vec = _mm256_loadu_ps(strike + i);
        __m256 rate_vec = _mm256_loadu_ps(rate + i);
        __m256 volatility_vec = _mm256_loadu_ps(volatility + i);
        __m256 otime_vec = _mm256_loadu_ps(otime + i);
        __m256 otype_vec = char_to_float(otype + i);

        __m256 result = blackScholes_simd(sptprice_vec, strike_vec, rate_vec, volatility_vec, otime_vec, otype_vec);

        _mm256_storeu_ps(output + i, result);
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

void * parallel(void* args){
    args_t* a = (args_t*) args;
    for (size_t i = 0; i < a->num_stocks; i++) {
        float sptPrice   = a->sptPrice  [i];
        float strike     = a->strike    [i];
        float rate       = a->rate      [i];
        float volatility = a->volatility[i];
        float otime      = a->otime     [i];
        char  otype_c     = a->otype     [i];

        float otype = (tolower(otype_c) == 'p')? 1 : 0;

        a->output[i] = blackScholes_mimd(sptPrice, strike, rate, volatility, otime, otype);
    }
    return NULL;
}

/* Alternative Implementation */
void* impl_parallel(void* args)
{

    args_t* a = (args_t*) args;
    /* Get all the arguments */
    register       float*   output = (float*)(a->output);
    register const float*   sptPrice = (const float*)(a->sptPrice);
    register const float*   strike = (const float*)(a->strike);
    register const float*   rate = (const float*)(a->rate);
    register const float*   volatility = (const float*)(a->volatility);
    register const float*   otime = (const float*)(a->otime);
    register const char*    otype = (const char*)(a->otype);
    register       size_t num_stocks = a->num_stocks;

    register       size_t nthreads = a->nthreads;
    register       size_t cpu      = a->cpu;

    /* Create all threads */
    pthread_t tid[nthreads];
    args_t    targs[nthreads];

    /* Assign current CPU to us */
    tid[0] = pthread_self();

    /* Amount of work per thread */
    size_t size_per_thread = num_stocks / nthreads;

    for (int i = 1; i < nthreads; i++) {
        /* Initialize the argument structure */
        targs[i].num_stocks = size_per_thread;
        targs[i].sptPrice   = (float*)(sptPrice + (i * size_per_thread));
        targs[i].strike     = (float*)(strike + (i * size_per_thread));
        targs[i].rate       = (float*)(rate + (i * size_per_thread));
        targs[i].volatility = (float*)(volatility + (i * size_per_thread));
        targs[i].otime      = (float*)(otime + (i * size_per_thread));
        targs[i].otype      = (char*)(otype + (i * size_per_thread));
        targs[i].output     = (float*)(output + (i * size_per_thread));
        targs[i].cpu      = (cpu + i) % nthreads;
        targs[i].nthreads = nthreads;
        pthread_create(&tid[i], NULL, parallel, (void*) &targs[i]);
    }

    /* Perform one portion of the work */
    for (size_t i = 0; i < size_per_thread; i++) {
        float sptPrice_i   = sptPrice  [i];
        float strike_i     = strike    [i];
        float rate_i       = rate      [i];
        float volatility_i = volatility[i];
        float otime_i      = otime     [i];
        char  otype_i      = otype     [i];

        float otype = (tolower(otype_i) == 'p')? 1 : 0;

        output[i] = blackScholes_mimd(sptPrice_i, strike_i, rate_i, volatility_i, otime_i, otype);
    }

    /* Perform trailing elements */
    int remaining = num_stocks % nthreads;
    for (size_t i = num_stocks - remaining; i < a->num_stocks; i++) {
        float sptPrice_i   = sptPrice  [i];
        float strike_i     = strike    [i];
        float rate_i       = rate      [i];
        float volatility_i = volatility[i];
        float otime_i      = otime     [i];
        char  otype_i      = otype     [i];

        float otype = (tolower(otype_i) == 'p')? 1 : 0;

        output[i] = blackScholes_mimd(sptPrice_i, strike_i, rate_i, volatility_i, otime_i, otype);
    }

    /* Join all threads */
    for (int i = 1; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }

    return NULL;
}
```

# 4. Results:
### The following table shows the results of the different implementations on different datasets on two different machines.

## machine one (11th Gen Intel(R) Core(TM) i7-1165G7 8 threads):
| Implementation | Time (ns) | Speedup | dataset |
|----------------|-----------|---------|---------|
| scalar         | 1208      | 1.0000  | dev     |
| vector         | 777       | 1.5547  | dev     |
| parallel       | 73510     | 0.0164  | dev     |
| scalar         | 186323    | 1.0000  | small   |
| vector         | 103662    | 1.7974  | small   |
| parallel       | 107077    | 1.7401  | small   |
| scalar         | 747813    | 1.0000  | medium  |
| vector         | 414325    | 1.8049  | medium  |
| parallel       | 218491    | 3.4226  | medium  |
| scalar         | 3023351   | 1.0000  | large   |
| vector         | 1690273   | 1.7887  | large   |
| parallel       | 705612    | 4.2847  | large   |
| scalar         | 471111120 | 1.0000  | native  |
| vector         | 266044477 | 1.7708  | native  |
| parallel       | 120517458 | 3.9091  | native  |

## machine two (AMD Ryzen 9 5900X 24 threads):
| Implementation | Time (ns) | Speedup | dataset |
|----------------|-----------|---------|---------|
| scalar         | 966       | 1.0000  | dev     |
| vector         | 434       | 2.2258  | dev     |
| parallel       | 506478    | 0.0019  | dev     |
| scalar         | 121257    | 1.0000  | small   |
| vector         | 71089     | 1.7057  | small   |
| parallel       | 539252    | 0.2248  | small   |
| scalar         | 480121    | 1.0000  | medium  |
| vector         | 282896    | 1.6971  | medium  |
| parallel       | 526600    | 0.9117  | medium  |
| scalar         | 1950527   | 1.0000  | large   |
| vector         | 1142842   | 1.7067  | large   |
| parallel       | 597773    | 3.2629  | large   |
| scalar         | 302045428 | 1.0000  | native  |
| vector         | 176733424 | 1.7090  | native  |
| parallel       | 21378723  | 14.1283 | native  |

formula for speedup: speedup = time of scalar implementation / time of implementation

As we can see from the results, the parallel implementation is the fastest implementation in large datasets.
The vector implementation is the fastest implementation in small datasets.
Parallel implementation didn't perform well in small datasets, but it performed well in large datasets.
That is because the overhead of creating threads is very high,
and it is not worth it in small datasets. Also, adding more threads will not always improve the performance.
Because the overhead of creating threads will be higher than the performance improvement.
The overhead of creating threads is tiny in large datasets, and it is worth it in large datasets.
The vector implementation performed well in small datasets,
but it didn't perform as good as parallel implementation in large datasets.
That is because the vector implementation is limited by the size of the vector registers.
The vector registers are 256 bits, and they can hold 8 floats.
I think if the vector implementation also vectorized the log function and exp function, it will perform even more better.
But I couldn't vectorize the log function and exp function because I did not have enough time.

# 4. Conclusion:
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

# 5. References
https://github.com/hawajkm/characterize-microbenchmark.git


# 6. Machine and OS Specifications

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