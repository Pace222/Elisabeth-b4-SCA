#ifndef MASKING_H
#define MASKING_H
#include "types.h"

#define N_SHARES 3

void gen_shares(uint4_t*);
void init_shares(uint4_t*, uint4_t);
uint4_t consume_shares(const uint4_t*);
void masked_addition(uint4_t*, const uint4_t*, const uint4_t*);
void masked_addition_constant(uint4_t*, const uint4_t*, uint4_t);
void masked_negation(uint4_t*, const uint4_t*);
void masked_sbox_second_order(uint4_t*, const uint4_t*, const uint4_t*);

#endif