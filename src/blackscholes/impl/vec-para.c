//
// Created by khormi on 1/2/24.
//

#include <stdio.h>
#include <ctype.h>
#include <pthread.h>
/* Include application-specific headers */
#include "../include/types.h"

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