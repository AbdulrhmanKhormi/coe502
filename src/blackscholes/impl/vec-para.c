//
// Created by khormi on 1/2/24.
//

#include <stdio.h>
#include <ctype.h>
#include <pthread.h>
/* Include application-specific headers */
#include "../include/types.h"

#include "vec.h"

void* impl_vector_para(void* args) {
    args_t* a = (args_t*) args;

    register float*   output = a->output;
    register float*   sptPrice = a->sptPrice;
    register float*   strike = a->strike;
    register float*   rate = a->rate;
    register float*   volatility = a->volatility;
    register float*   otime = a->otime;
    register char*    otype = a->otype;
    register size_t   num_stocks = a->num_stocks;
    register size_t   nthreads = a->nthreads;

    pthread_t tid[nthreads];
    args_t    targs[nthreads];

    tid[0] = pthread_self();

    size_t size_per_thread = num_stocks / nthreads;

    for (int i = 1; i < nthreads; i++) {
        targs[i].num_stocks = size_per_thread;
        targs[i].sptPrice   = (sptPrice + (i * size_per_thread));
        targs[i].strike     = (strike + (i * size_per_thread));
        targs[i].rate       = (rate + (i * size_per_thread));
        targs[i].volatility = (volatility + (i * size_per_thread));
        targs[i].otime      = (otime + (i * size_per_thread));
        targs[i].otype      = (otype + (i * size_per_thread));
        targs[i].output     = (output + (i * size_per_thread));
        pthread_create(&tid[i], NULL, impl_vector, &targs[i]);
    }

    for (size_t i = 0; i < size_per_thread; i+=8) {
        __m256 sptprice_vec = _mm256_loadu_ps(a->sptPrice + i);
        __m256 strike_vec = _mm256_loadu_ps(a->strike + i);
        __m256 rate_vec = _mm256_loadu_ps(a->rate + i);
        __m256 volatility_vec = _mm256_loadu_ps(a->volatility + i);
        __m256 otime_vec = _mm256_loadu_ps(a->otime + i);
        __m256 otype_vec = char_to_float(a->otype + i);

        __m256 result = blackScholes_simd(sptprice_vec, strike_vec, rate_vec, volatility_vec, otime_vec, otype_vec);

        _mm256_storeu_ps(a->output + i, result);
    }

    int remaining = num_stocks % nthreads;
    for (size_t i = num_stocks - remaining; i < a->num_stocks; i+=8) {
        __m256 sptprice_vec = _mm256_loadu_ps(a->sptPrice + i);
        __m256 strike_vec = _mm256_loadu_ps(a->strike + i);
        __m256 rate_vec = _mm256_loadu_ps(a->rate + i);
        __m256 volatility_vec = _mm256_loadu_ps(a->volatility + i);
        __m256 otime_vec = _mm256_loadu_ps(a->otime + i);
        __m256 otype_vec = char_to_float(a->otype + i);

        __m256 result = blackScholes_simd(sptprice_vec, strike_vec, rate_vec, volatility_vec, otime_vec, otype_vec);

        _mm256_storeu_ps(a->output + i, result);
    }

    for (int i = 1; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
    }

    return NULL;

}