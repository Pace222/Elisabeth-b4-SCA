#include "masking.h"
#include <stdlib.h>

packed gen_shares() {
    return rand() & MASK_TOT;
}

packed init_shares(uint4_t value) {
    packed out_shares = gen_shares();
    return (out_shares & MASK_0_1) | ((((value - SHARE_0(out_shares)) & MASK_2) - SHARE_1(out_shares)) & MASK_2);
}

uint4_t consume_shares(packed shares) {
    return uint4_add(uint4_add(SHARE_0(shares), SHARE_1(shares)), SHARE_2(shares));
}

packed masked_sbox_second_order(packed inp_shares, const uint4_t* s_box) {
    uint4_t r = uint4_new(gen_shares());
    uint4_t r_prime = uint4_add(r, uint4_add(SHARE_0(inp_shares), SHARE_1(inp_shares)));

    packed output_shares[0x10];

    for (int a = 0; a < 0x10; ++a) {
        output_shares[uint4_add(a, r_prime)] = init_shares(s_box[uint4_add(SHARE_2(inp_shares), uint4_neg(a))]);
    }

    return output_shares[r];
}

