#include "masking.h"
#include <stdlib.h>

int gen_rand() {
    return rand();
}

void gen_shares(uint4_t* out_shares) {
    for (int i = 0; i < N_SHARES - 1; i++) {
        out_shares[i] = uint4_new(gen_rand());
    }
}

void init_shares(uint4_t* out_shares, uint4_t value) {
    gen_shares(out_shares + 1);

    uint4_t rest = uint4_new(0);
    for (int i = 1; i < N_SHARES; i++) {
        rest = uint4_add(rest, out_shares[i]);
    }
    out_shares[0] = uint4_add(value, uint4_neg(rest));
}

uint4_t consume_shares(const uint4_t* shares) {
    uint4_t res = uint4_new(0);
    for (int i = 0; i < N_SHARES; i++) {
        res = uint4_add(res, shares[i]);
    }
    return res;
}

void masked_addition(uint4_t* output_shares, const uint4_t* inp1_shares, const uint4_t* inp2_shares) {
    for (int i = 0; i < N_SHARES; i++) {
        output_shares[i] = uint4_add(inp1_shares[i], inp2_shares[i]);
    }
}

void masked_addition_constant(uint4_t* output_shares, const uint4_t* inp1_shares, uint4_t inp2_constant) {
    output_shares[0] = uint4_add(inp1_shares[0], inp2_constant);
    for (int i = 1; i < N_SHARES; i++) {
        output_shares[i] = inp1_shares[i];
    }
}

void masked_negation(uint4_t* output_shares, const uint4_t* inp_shares) {
    for (int i = 0; i < N_SHARES; i++) {
        output_shares[i] = uint4_neg(inp_shares[i]);
    }
}

void masked_sbox_second_order(uint4_t* output_shares, const uint4_t* inp_shares, const uint4_t* s_box) {
    uint4_t t[0x10];
    uint4_t r = uint4_new(gen_rand());
    uint4_t r_prime = uint4_add(r, uint4_add(inp_shares[1], inp_shares[2]));

    gen_shares(output_shares + 1);

    for (int a = 0; a < 0x10; ++a) {
        t[uint4_add(a, r_prime)] = uint4_add(uint4_add(s_box[uint4_add(inp_shares[0], uint4_neg(a))], uint4_neg(output_shares[1])), uint4_neg(output_shares[2]));
    }

    output_shares[0] = t[r];
}

