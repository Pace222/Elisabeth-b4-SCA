#ifndef GENERATOR_CHA_H
#define GENERATOR_CHA_H

#include <stddef.h>

#include "generator.h"
#include "chacha.h"
#include "filtering_4.h"
#include "filtering_b4.h"

typedef struct {
    rng r;
    ECRYPT_ctx ctx;
    size_t batch_idx;
} rng_cha;

void rng_new_cha(rng_cha*, const uint8_t*, int);

#endif