/* para.c
 *
 * Author:
 * Date  :
 *
 *  Description
 */

/* Standard C includes */
#include <stdlib.h>

/* Include common headers */
#include "../../common/types.h"
#include "../../common/macros.h"

/* Include application-specific headers */
#include "../include/types.h"

// using pthreads for blackscholes alg


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include <sched.h>

#define inv_sqrt_2xPI 0.39894228040143270286

float CNDF_mimd(float InputX){
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
float blackScholes_mimd(float sptprice, float strike, float rate, float volatility, float otime, int otype){
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

    NofXd1 = CNDF_mimd(d1);
    NofXd2 = CNDF_mimd(d2);

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
