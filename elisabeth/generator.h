#ifndef GENERATOR_H
#define GENERATOR_H

#include <stddef.h>

#include "filtering_4.h"
#include "filtering_b4.h"

typedef struct _rng rng;

struct _rng {
    uint16_t indices[KEY_WIDTH_B4];
    uint4_t whitening[KEYROUND_WIDTH_B4];
    int mode;
    uint8_t (*gen_rand_uniform)(rng*, uint8_t*);
    void (*copy)(rng*, const rng*);
    void (*next_elem)(rng*);
};

void switch_endianness(uint8_t*, const uint8_t*, int);

#endif