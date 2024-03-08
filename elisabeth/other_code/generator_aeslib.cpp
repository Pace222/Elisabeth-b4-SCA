#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <Profiler.h>

#include "generator.h"

AESLib aes;
byte NULL_IV[N_BLOCK] = { 0 };

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
    for (byte* b = r->batch; b < r->batch + BATCH_SIZE; b += N_BLOCK) {
        aes.encrypt(r->ctr, N_BLOCK, b, r->key, N_BLOCK, NULL_IV);
        increment_with_carry(r->ctr, N_BLOCK);
    }
    r->batch_idx = 0;
}

void rng_new(rng* r, uint8_t* seed, int mode) {
    int KEY_WIDTH = (mode ? KEY_WIDTH_4 : KEY_WIDTH_B4);
    size_t* indices = (mode ? r->indices._4 : r->indices._b4);

    switch_endianness(r->key, seed, N_BLOCK);
    memset(r->ctr, 0, N_BLOCK);
    generate_batch(r);

    r->mode = mode;
    for (int i = 0; i < KEY_WIDTH; i++) {
        indices[i] = i;
    }
}

uint8_t random_uniform(rng* r) {
    if (r->batch_idx >= BATCH_SIZE) {
        {
            profiler_t p;
            generate_batch(r);
        }
    }

    return r->batch[r->batch_idx++];
}

size_t random_uniform_n_lsb(rng* r, size_t n) {
    uint32_t random_u32 = 0;
    for (int i = 0; i < 4; i++) {
        random_u32 |= (random_uniform(r) << (i * 8));
    }

    return random_u32 >> (32 - n);
}

size_t gen_range(rng* r, size_t min, size_t max) {
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
