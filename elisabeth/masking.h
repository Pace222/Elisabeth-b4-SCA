#ifndef MASKING_H
#define MASKING_H
#include <stdlib.h>

#include "types.h"

/* 2-share arithmetic shares format: ____________________AAAA____BBBB (32-bit word with 2 shares) */
typedef uint32_t masked;
#define MASK_0   0b0000111100000000
#define MASK_1   0b0000000000001111
#define MASK_TOT 0b0000111100001111
#define SHARE_0(shares) (((shares) & MASK_0) >> 8)
#define SHARE_1(shares) (((shares) & MASK_1))

/* Methods for mask shares manipulation */
#define masked_addition(inp1_shares, inp2_shares) (((inp1_shares) + (inp2_shares)) & MASK_TOT)
#define masked_addition_constant(inp1_shares, inp2_constant) (((inp1_shares) + (inp2_constant)) & MASK_TOT)
#define masked_negation(inp_shares) (((~(inp_shares) & MASK_TOT) + 0b0000000100000001) & MASK_TOT)

uint32_t get_rand();
uint4_t consume_shares(masked);
void init_shares(masked*, uint4_t*, size_t);
void init_null_shares(masked*, uint4_t*, size_t);
void generate_masking_random_table();
void generate_null_masking_random_table();
void generate_shuffling_random_table();
void generate_masking_shuffling_random_table();
void reset_masking_counter();
masked masked_sbox_first_order(masked, const uint8_t[16][16]);

#endif