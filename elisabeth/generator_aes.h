#ifndef GENERATOR_AES_H
#define GENERATOR_AES_H

#include <stddef.h>

#include "generator.h"
#include "aes.h"

typedef struct {
    rng r;                                     /* "Child class" of rng */
    struct AES_ctx ctx;                        /* AES context */
    uint8_t ctr[AES_BLOCKLEN];                 /* AES-CTR */
    size_t batch_idx;                          /* Head inside batch */
} rng_aes;

void rng_new_aes(rng_aes*, const uint8_t*, int);

#endif