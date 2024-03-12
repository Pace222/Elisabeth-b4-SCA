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

uint32_t random_uniform_n_lsb(rng* r, size_t n, uint8_t* batch) {
    uint32_t random_u32 = 0;
    for (int i = 0; i < 4; i++) {
        random_u32 |= (r->gen_rand_uniform(r, batch) << (i * 8));
    }

    return random_u32 >> (32 - n);
}

uint32_t gen_range(rng* r, size_t min, size_t max, uint8_t* batch) {
    if (min > max) {
        return -1;
    }

    size_t bit_len = (size_t) floor(log2(max - min));
    size_t a = min + random_uniform_n_lsb(r, bit_len, batch);

    while (a >= max) {
        a = min + random_uniform_n_lsb(r, bit_len, batch);
    }
    return a;
}

void precompute_prng(rng* r, uint8_t* batch) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int KEY_WIDTH = r->mode ? KEY_WIDTH_4 : KEY_WIDTH_B4;

    // Select a random subset without repetition of size KEYROUND_WIDTH from the key
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        uint32_t j = gen_range(r, i, KEY_WIDTH, batch);
        size_t tmp = r->indices[i];
        r->indices[i] = r->indices[j];
        r->indices[j] = tmp;
    }
    
    // Generate a random whitening mask
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        r->whitening[i] = uint4_new(r->gen_rand_uniform(r, batch));
    }
}

void copy_rng(rng* dest, const rng* src) {
    memcpy(dest->indices, src->indices, sizeof(src->indices));
    memcpy(dest->whitening, src->whitening, sizeof(src->whitening));
    dest->mode = src->mode;
    dest->gen_rand_uniform = src->gen_rand_uniform;
    dest->copy = src->copy;
    dest->next_elem = src->next_elem;
}

void rng_new(rng* r, int mode, uint8_t (*gen_rand_uniform)(rng*, uint8_t*), void (*copy)(rng*, const rng*), void (*next_elem)(rng*)) {
    int KEY_WIDTH = mode ? KEY_WIDTH_4 : KEY_WIDTH_B4;

    for (int i = 0; i < KEY_WIDTH; i++) {
        r->indices[i] = i;
    }
    r->mode = mode;
    r->gen_rand_uniform = gen_rand_uniform;
    r->copy = copy;
    r->next_elem = next_elem;
}
