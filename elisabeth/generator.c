#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "generator.h"

void switch_endianness(uint8_t* dest, const uint8_t* src, int length) {
    for (int i = 0; i < length; i++) {
        dest[length - 1 - i] = src[i];
    }
}

void increment_with_carry(uint8_t* array, size_t length) {
    int carry = 1;

    for (int i = 0; i < length; i++) {
        int sum = array[i] + carry;
        array[i] = sum & 0xFF;
        carry = sum >> 8;
        if (carry == 0) break;
    }
}

void generate_batch(rng* r) {
    for (uint8_t* b = r->batch; b < r->batch + BATCH_SIZE; b += AES_BLOCKLEN) {
        memcpy(b, r->ctr, AES_BLOCKLEN);
        AES_ECB_encrypt(&r->ctx, b);
        increment_with_carry(r->ctr, AES_BLOCKLEN);
    }
    r->batch_idx = 0;
}

void rng_new(rng* r, uint8_t* seed_little_end, int mode) {
    int KEY_WIDTH = (mode ? KEY_WIDTH_4 : KEY_WIDTH_B4);
    size_t* indices = (mode ? r->indices._4 : r->indices._b4);

    AES_init_ctx(&r->ctx, seed_little_end);
    memset(r->ctr, 0, AES_BLOCKLEN);
    generate_batch(r);

    r->mode = mode;
    for (int i = 0; i < KEY_WIDTH; i++) {
        indices[i] = i;
    }
}

uint8_t random_uniform(rng* r) {
    if (r->batch_idx >= BATCH_SIZE) {
        generate_batch(r);
    }

    return r->batch[r->batch_idx++];
}

uint32_t random_uniform_n_lsb(rng* r, size_t n) {
    uint32_t random_u32 = 0;
    for (int i = 0; i < 4; i++) {
        random_u32 |= (random_uniform(r) << (i * 8));
    }

    return random_u32 >> (32 - n);
}

uint32_t gen_range(rng* r, size_t min, size_t max) {
    if (min > max) {
        return -1;
    }

    size_t bit_len = (size_t) floor(log2(max - min));
    size_t a = min + random_uniform_n_lsb(r, bit_len);

    while (a >= max) {
        a = min + random_uniform_n_lsb(r, bit_len);
    }
    return a;
}

void precompute_prng(uint32_t* dest, rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int KEY_WIDTH = r->mode ? KEY_WIDTH_4 : KEY_WIDTH_B4;

    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        dest[i] = gen_range(r, i, KEY_WIDTH);
    }
    for (int i = KEYROUND_WIDTH; i < 2 * KEYROUND_WIDTH; i++) {
        dest[i] = uint4_new(random_uniform(r));
    }
}
