#ifndef GENERATOR_AES_H
#define GENERATOR_AES_H

#include <stddef.h>

#include "generator.h"
#include "aes.h"
#include "filtering_4.h"
#include "filtering_b4.h"

typedef struct {
    rng r;
    struct AES_ctx ctx;
    uint8_t ctr[AES_BLOCKLEN];
    size_t batch_idx;
} rng_aes;

void rng_new_aes(rng_aes*, const uint8_t*, int);

#endif