#ifndef GENERATOR_H
#define GENERATOR_H

#include <stddef.h>

#include "aes.h"
#include "filtering_4.h"
#include "filtering_b4.h"

#define BATCH_SIZE 128

typedef struct {
    struct AES_ctx ctx;
    uint8_t ctr[AES_BLOCKLEN];
    uint8_t batch[BATCH_SIZE];
    uint16_t batch_idx;
    int mode;
    size_t indices[KEY_WIDTH_B4];
} rng;

void switch_endianness(uint8_t*, const uint8_t*, int);
void rng_new(rng*, const uint8_t*, int);
void rng_reset_indices(rng*);
uint8_t random_uniform(rng*);
uint32_t gen_range(rng*, size_t, size_t);
void precompute_prng(uint32_t*, rng*);

#endif