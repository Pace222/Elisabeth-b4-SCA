#ifndef MASKING_H
#define MASKING_H
#include "types.h"

#define N_SHARES 3

void gen_n_shares(uint4_t*, unsigned int);
uint4_t consume_n_shares(uint4_t*, unsigned int);
void masked_addition_second_order(uint4_t*, uint4_t*, uint4_t*);
void masked_sbox_second_order(uint4_t*, uint4_t*, uint4_t*);

#endif