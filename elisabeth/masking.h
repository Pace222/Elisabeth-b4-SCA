#ifndef MASKING_H
#define MASKING_H
#include <stdlib.h>

#include "types.h"

// Packed additive shares format: ____________________AAAA____BBBB (32-bit word with 2 shares)
typedef uint32_t add_packed;
#define MASK_0   0b0000111100000000
#define MASK_1   0b0000000000001111
#define MASK_TOT 0b0000111100001111
#define SHARE_0(shares) (((shares) & MASK_0) >> 8)
#define SHARE_1(shares) (((shares) & MASK_1))

// Packed multiplicative shares format: AAAA_______________________BBBBB (32-bit word with 1 share and 1 multiplicative share)
typedef uint32_t mul_packed;
#define MUL_MODULO 17
#define MUL_MASK_0 0b11110000000000000000000000000000
#define MUL_MASK_1 0b00000000000000000000000000011111
#define MUL_SHARE_0(shares) (((shares) & MUL_MASK_0) >> 28)
#define MUL_SHARE_1(shares) (((shares) & MUL_MASK_1))

#define masked_addition(inp1_shares, inp2_shares) (((inp1_shares) + (inp2_shares)) & MASK_TOT)
#define masked_addition_constant(inp1_shares, inp2_constant) (((inp1_shares) + (inp2_constant)) & MASK_TOT)
#define masked_negation(inp_shares) (((~(inp_shares) & MASK_TOT) + 0b0000000100000001) & MASK_TOT)

uint32_t get_rand();
uint4_t mul_consume_shares(mul_packed);
uint4_t consume_shares(add_packed);
void init_shares(add_packed*, uint4_t*, size_t);
void init_null_shares(add_packed*, uint4_t*, size_t);
void generate_masking_random_table();
void generate_null_masking_random_table();
void generate_shuffling_random_table();
void generate_masking_shuffling_random_table();
void reset_counter();
add_packed masked_sbox_first_order(add_packed, const uint32_t*);
//packed masked_sbox_second_order(packed, const uint32_t*);

#endif