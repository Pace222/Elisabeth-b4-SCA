#ifndef MASKING_H
#define MASKING_H
#include "types.h"

// Packed shares format: __________________AAAA_BBBB_CCCC (32-bit word with 3 shares)
typedef uint32_t packed;
#define MASK_0   0b011110000000000
#define MASK_1   0b000000111100000
#define MASK_2   0b000000000001111
#define MASK_0_1 0b011110111100000
#define MASK_TOT 0b011110111101111
#define SHARE_0(shares) (((shares) & MASK_0) >> 10)
#define SHARE_1(shares) (((shares) & MASK_1) >> 5)
#define SHARE_2(shares) (((shares) & MASK_2))

#define masked_addition(inp1_shares, inp2_shares) (((inp1_shares) + (inp2_shares)) & MASK_TOT)
#define masked_addition_constant(inp1_shares, inp2_constant) (((inp1_shares) + (inp2_constant)) & MASK_TOT)
#define masked_negation(inp_shares) (((~(inp_shares) & MASK_TOT) + 0b000010000100001) & MASK_TOT)

packed gen_rand();
void generate_random_table();
packed init_shares(uint4_t);
uint4_t consume_shares(packed);
packed masked_sbox_second_order(packed, const uint32_t*);

#endif