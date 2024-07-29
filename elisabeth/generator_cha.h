#ifndef GENERATOR_CHA_H
#define GENERATOR_CHA_H

#include <stddef.h>

#include "generator.h"
#include "chacha.h"

typedef struct {
    rng r;                                     /* "Child class" of rng */
    ECRYPT_ctx ctx;                            /* ChaCha20 context */
    size_t batch_idx;                          /* Head inside batch */
} rng_cha;

void rng_new_cha(rng_cha*, const uint8_t*, int);

#endif