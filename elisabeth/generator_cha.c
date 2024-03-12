#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "generator_cha.h"

#define CHACHA_BATCH_SIZE 64 * (1 + 5 * KEYROUND_WIDTH_4 / 64)

void new_batch_cha(uint8_t* batch, rng_cha* r_cha) {
    ECRYPT_keystream_bytes(&r_cha->ctx, batch, CHACHA_BATCH_SIZE);
}

uint8_t random_uniform_cha(rng* r, uint8_t* batch) {
    rng_cha* r_cha = (rng_cha*) r;

    if (r_cha->batch_idx >= CHACHA_BATCH_SIZE) {
        new_batch_cha(batch, r_cha);
        r_cha->batch_idx = 0;
    }

    return batch[r_cha->batch_idx++];
}

void copy_cha(rng* dest, const rng* src) {
    rng_cha* dest_cha = (rng_cha*) dest;
    rng_cha* src_cha = (rng_cha*) src;

    copy_rng(&dest_cha->r, &src_cha->r);
    
    memcpy(&dest_cha->ctx, &src_cha->ctx, sizeof(src_cha->ctx));
    dest_cha->batch_idx = src_cha->batch_idx;
}

void next_elem_cha(rng* r) {
    rng_cha* r_cha = (rng_cha*) r;

    uint8_t batch[CHACHA_BATCH_SIZE];
    new_batch_cha(batch, r_cha);
    r_cha->batch_idx = 0;

    precompute_prng(&r_cha->r, batch);
}

void rng_new_cha(rng_cha* r_cha, const uint8_t* seed_little_end, int mode) {

    rng_new(&r_cha->r, mode, random_uniform_cha, copy_cha, next_elem_cha);
    
    ECRYPT_init_ctx(&r_cha->ctx, seed_little_end);

    uint8_t batch[CHACHA_BATCH_SIZE];
    new_batch_cha(batch, r_cha);
    r_cha->batch_idx = 0;

    precompute_prng(&r_cha->r, batch);
}