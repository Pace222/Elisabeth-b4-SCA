#include "masking.h"
#include <stdlib.h>

#define RANDOM_TABLE_SIZE 2048
packed RANDOM_TABLE[RANDOM_TABLE_SIZE];
size_t random_table_idx = 0;
//#define gen_shares() (RANDOM_TABLE[random_table_idx++] & MASK_TOT)

packed gen_shares() {
    return RANDOM_TABLE[random_table_idx++] & MASK_TOT;
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

packed masked_sbox_second_order(packed inp_shares, const uint4_t* s_box) {
    uint4_t t[0x10];
    uint4_t r = uint4_new(gen_shares());
    uint4_t r_prime = uint4_add(r, uint4_neg(uint4_add(SHARE_0(inp_shares), SHARE_1(inp_shares))));

    packed output_shares = gen_shares();
    uint4_t neg_out_summed = uint4_neg(uint4_add(SHARE_0(output_shares), SHARE_1(output_shares)));
    uint4_t inp_2 = SHARE_2(inp_shares);

    //uint4_t neg_out_summed_32 = (neg_out_summed_4 << 25) | (neg_out_summed_4 << 20) | (neg_out_summed_4 << 15) | (neg_out_summed_4 << 10) | (neg_out_summed_4 << 5) | neg_out_summed_4;
    /*for (int a = 0; a < 3; ++a) {
        t[uint4_add(a, r_prime)] = (s_box[uint4_add(inp_2, a)] + neg_out_summed_32) & 0b00011110111101111011110111101111;
    }*/

    for (int a = 0; a < 0x10; ++a) {
        t[uint4_add(a, r_prime)] = uint4_add(s_box[uint4_add(inp_2, a)], neg_out_summed);
    }
    return (output_shares & MASK_0_1) | t[r];
}

void random_permutation(size_t* out_array, size_t n) {
    // initial range of numbers
    for(int i = 0; i < n; ++i){
        out_array[i] = i + 1;
    }

    for (int i = n - 1; i >= 0; --i){
        //generate a random number [0, i]
        int j = gen_shares() % (i + 1);

        //swap the last element with element at random index
        int temp = out_array[i];
        out_array[i] = out_array[j];
        out_array[j] = temp;
    }
}