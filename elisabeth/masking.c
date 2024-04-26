#include "masking.h"
#include <stdlib.h>

#define RANDOM_TABLE_SIZE 2048
packed RANDOM_TABLE[RANDOM_TABLE_SIZE];
size_t random_table_idx = 0;

uint32_t gen_rand() {
    return RANDOM_TABLE[random_table_idx++];
}

packed gen_shares() {
    return gen_rand() & MASK_TOT;
} 

void generate_random_table() {
    for (size_t i = 0; i < RANDOM_TABLE_SIZE; i++) {
        RANDOM_TABLE[i] = rand();
    }
    random_table_idx = 0;
}

packed init_shares(uint4_t value) {
    packed out_shares = gen_shares();
    return (out_shares & MASK_0_1) | ((((value - SHARE_0(out_shares)) & MASK_2) - SHARE_1(out_shares)) & MASK_2);
}

uint4_t consume_shares(packed shares) {
    return uint4_add(uint4_add(SHARE_0(shares), SHARE_1(shares)), SHARE_2(shares));
}

packed masked_sbox_second_order(packed inp_shares, const uint32_t* s_box) {
    uint32_t t[4] = { 0 };
    uint4_t r = uint4_new(gen_rand());
    uint4_t r_prime = uint4_add(r, uint4_neg(uint4_add(SHARE_0(inp_shares), SHARE_1(inp_shares))));
    uint4_t off_left = 5 * (r_prime % 4), off_right = 20 - off_left;

    packed output_shares = gen_shares();
    uint4_t neg_out_summed_4 = uint4_neg(uint4_add(SHARE_0(output_shares), SHARE_1(output_shares)));
    uint32_t neg_out_summed_32 = (neg_out_summed_4 << 15) | (neg_out_summed_4 << 10) | (neg_out_summed_4 << 5) | neg_out_summed_4;
    uint4_t inp_2 = SHARE_2(inp_shares);
    
    for (uint4_t a = 0, curr = r_prime / 4; a < 0x10; a += 4, curr = (curr + 1) % 4) {
        uint32_t s = (s_box[uint4_add(inp_2, a) / 4] + neg_out_summed_32);
        t[curr] |= (s >> off_left);
        t[(curr + 1) % 4] |= (s << off_right) & 0b01111011110111101111;
    }

    uint4_t idx = uint4_add(r, inp_2 % 4);
    return (output_shares & MASK_0_1) | uint4_new(t[idx / 4] >> (5 * (3 - (idx % 4))));
}
