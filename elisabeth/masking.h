#ifndef MASKING_H
#define MASKING_H
#include <stdlib.h>

#include "types.h"

// Packed shares format: _______________________AAAA_BBBB (32-bit word with 2 shares)
typedef uint32_t packed;
#define MASK_0   0b0111100000
#define MASK_1   0b0000001111
#define MASK_TOT 0b0111101111
#define SHARE_0(shares) (((shares) & MASK_0) >> 5)
#define SHARE_1(shares) (((shares) & MASK_1))

#define masked_addition(inp1_shares, inp2_shares) (((inp1_shares) + (inp2_shares)) & MASK_TOT)
#define masked_addition_constant(inp1_shares, inp2_constant) (((inp1_shares) + (inp2_constant)) & MASK_TOT)
#define masked_negation(inp_shares) (((~(inp_shares) & MASK_TOT) + 0b0000100001) & MASK_TOT)

uint32_t get_rand();
uint4_t consume_shares(packed);
void init_shares(packed*, uint4_t*, size_t);
void init_null_shares(packed*, uint4_t*, size_t);
void generate_masking_random_table();
void generate_null_masking_random_table();
void generate_shuffling_random_table();
void generate_masking_shuffling_random_table();
void reset_counter();
packed masked_sbox_first_order(packed, const uint32_t*);
//packed masked_sbox_second_order(packed, const uint32_t*);

#endif