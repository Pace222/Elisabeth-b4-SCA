#ifndef GENERATOR_H
#define GENERATOR_H

#include <stddef.h>

#include "types.h"

typedef struct _rng rng;

struct _rng {
    uint16_t indices[KEY_WIDTH_B4];            /* Selected indices in the key */
    uint4_t whitening[KEYROUND_WIDTH_B4];      /* Whitening of the keyround*/
    int mode;                                  /* 1 for Elisabeth-4, 0 for Elisabeth-b4*/
    uint8_t (*gen_rand_uniform)(rng*, uint8_t*);   /* "Abstract method" that returns a random number from a PRNG context and batch */
    void (*copy)(rng*, const rng*);            /* "Abstract method" specifying how to copy a PRNG context */
    void (*next_elem)(rng*);                   /* "Abstract method" specifying how to compute the next element from the PRNG context */
};

void switch_endianness(uint8_t*, const uint8_t*, int);
void precompute_prng(rng*, uint8_t*);
void copy_rng(rng*, const rng*);
void rng_new(rng*, int, uint8_t (*)(rng*, uint8_t*), void (*)(rng*, const rng*), void (*)(rng*));

#endif