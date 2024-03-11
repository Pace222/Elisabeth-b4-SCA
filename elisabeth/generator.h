#ifndef GENERATOR_H
#define GENERATOR_H

#include <stddef.h>

#include "aes.h"
#include "filtering_4.h"
#include "filtering_b4.h"

#define BATCH_SIZE AES_BLOCKLEN * (1 + 5 * KEYROUND_WIDTH_4 / AES_BLOCKLEN)

typedef struct _rng rng;

struct _rng {
    uint16_t indices[KEY_WIDTH_B4];
    uint4_t whitening[KEYROUND_WIDTH_B4];
    int mode;
    uint8_t (*gen_rand_uniform)(rng*, uint8_t*);
    void (*copy)(rng*, const rng*);
    void (*next_elem)(rng*);
};

typedef struct {
    rng r;
    struct AES_ctx ctx;
    const uint8_t* key; // AES_KEYLEN
    uint8_t ctr[AES_BLOCKLEN];
    size_t batch_idx;
} rng_aes;

//typedef rng rng_chacha;

void switch_endianness(uint8_t*, const uint8_t*, int);
void rng_new_aes(rng_aes*, const uint8_t*, int);
//void rng_new_chacha(rng_chacha*, const uint8_t*, int);

#endif