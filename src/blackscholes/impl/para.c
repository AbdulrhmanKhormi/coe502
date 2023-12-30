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
    float* output = (float*) malloc(sizeof(float) * a->num_stocks);
    for (size_t i = 0; i < a->num_stocks; i++) {
        float sptPrice   = a->sptPrice  [i];
        float strike     = a->strike    [i];
        float rate       = a->rate      [i];
        float volatility = a->volatility[i];
        float otime      = a->otime     [i];
        char  otype_c     = a->otype     [i];

        float otype = (tolower(otype_c) == 'p')? 1 : 0;

        output[i] = blackScholes_mimd(sptPrice, strike, rate, volatility, otime, otype);
    }
    a->output = output;
    return NULL;
}

/* Alternative Implementation */
void* impl_parallel(void* args)
{

    args_t* a = (args_t*) args;
    int NUM_THREADS = a->nthreads;

    pthread_t threads[NUM_THREADS];
    args_t* args_array[NUM_THREADS];

    size_t i;
    for (i = 0; i < NUM_THREADS; i++) {
        args_array[i] = (args_t*) malloc(sizeof(args_t));
        args_array[i]->num_stocks = a->num_stocks / NUM_THREADS;
        args_array[i]->sptPrice = a->sptPrice + i * args_array[i]->num_stocks;
        args_array[i]->strike = a->strike + i * args_array[i]->num_stocks;
        args_array[i]->rate = a->rate + i * args_array[i]->num_stocks;
        args_array[i]->volatility = a->volatility + i * args_array[i]->num_stocks;
        args_array[i]->otime = a->otime + i * args_array[i]->num_stocks;
        args_array[i]->otype = a->otype + i * args_array[i]->num_stocks;
        pthread_create(&threads[i], NULL, parallel, (void*) args_array[i]);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    float* output = (float*) malloc(sizeof(float) * a->num_stocks);
    for (i = 0; i < NUM_THREADS; i++) {
        memcpy(output + i * args_array[i]->num_stocks, args_array[i]->output, sizeof(float) * args_array[i]->num_stocks);
    }
    a->output = output;

//    for (int j = 0; j < a->num_stocks; ++j) {
//        printf("%f\n", a->output[j]);
//    }
    return NULL;
}
