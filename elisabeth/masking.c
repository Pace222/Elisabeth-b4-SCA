#include "masking.h"
#include <stdio.h>

#define RANDOM_TABLE_SIZE 2048
packed RANDOM_TABLE[RANDOM_TABLE_SIZE];
size_t random_table_idx = 0;

#define shares_from_value(rand_shares, target_value) (((rand_shares) & MASK_0) | (((target_value) - SHARE_0((rand_shares))) & MASK_1))

uint32_t get_rand() {
    return RANDOM_TABLE[random_table_idx++];
}

uint4_t consume_shares(packed shares) {
    return uint4_add(SHARE_0(shares), SHARE_1(shares));
}

uint32_t printed_shares(uint32_t s) {
    printf("%X", SHARE_0(s));
    printf("%X", SHARE_1(s));
    return s;
}

uint32_t printed_rand(uint32_t r) {
    printf("%02X", r);
    return r;
}

void init_shares(packed* packed_values, uint4_t* values, size_t length) {
    for (size_t i = 0; i < length; i++) {
        packed shares = rand();
        packed_values[i] = printed_shares(shares_from_value(shares, values[i]));
    }
    printf("|");
    fflush(stdout);
}

void init_null_shares(packed* packed_values, uint4_t* values, size_t length) {
    for (size_t i = 0; i < length; i++) {
        packed_values[i] = printed_shares(values[i]);
    }
    printf("|");
    fflush(stdout);
}

void generate_masking_random_table() {
    uint32_t r = rand();
    RANDOM_TABLE[0] = shares_from_value(r, 0);
    size_t k = 1;
    for (size_t i = 0; i < 14; i++) {
        for (size_t j = 0; j < 18; j++) {
            RANDOM_TABLE[k++] = uint4_new(rand());
            RANDOM_TABLE[k++] = rand() & MASK_TOT;
        }
    }
    reset_counter();
}

void generate_null_masking_random_table() {
    RANDOM_TABLE[0] = 0;
    size_t k = 1;
    for (size_t i = 0; i < 14; i++) {
        for (size_t j = 0; j < 18; j++) {
            RANDOM_TABLE[k++] = 0;
            RANDOM_TABLE[k++] = 0;
        }
    }
    reset_counter();
}

void generate_shuffling_random_table() {
    RANDOM_TABLE[0] = printed_rand(rand() % 98);
    RANDOM_TABLE[1] = printed_rand(rand() % 14);
    size_t k = 2;
    for (size_t i = 0; i < 14; i++) {
        // Filter blocks
        RANDOM_TABLE[k++] = printed_rand(rand() % 3);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        
        RANDOM_TABLE[k++] = printed_rand(rand() % 3);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        
        RANDOM_TABLE[k++] = printed_rand(rand() % 2);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
    }
    printf("|");
    fflush(stdout);

    reset_counter();
}

void generate_masking_shuffling_random_table() {
    RANDOM_TABLE[0] = printed_rand(rand() % 98);
    uint32_t r = rand();
    RANDOM_TABLE[1] = shares_from_value(r, 0);
    RANDOM_TABLE[2] = printed_rand(rand() % 14);
    size_t k = 3;
    for (size_t i = 0; i < 14; i++) {
        // Filter blocks
        RANDOM_TABLE[k++] = printed_rand(rand() % 3);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        for (int j = 0; j < 6; j++) { // First S-box round
            RANDOM_TABLE[k++] = uint4_new(rand());
            RANDOM_TABLE[k++] = rand() & MASK_TOT;
        }
        RANDOM_TABLE[k++] = printed_rand(rand() % 3);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        for (int j = 0; j < 6; j++) { // First S-box round
            RANDOM_TABLE[k++] = uint4_new(rand());
            RANDOM_TABLE[k++] = rand() & MASK_TOT;
        }
        RANDOM_TABLE[k++] = printed_rand(rand() % 2);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        for (int j = 0; j < 6; j++) { // First S-box round
            RANDOM_TABLE[k++] = uint4_new(rand());
            RANDOM_TABLE[k++] = rand() & MASK_TOT;
        }
    }
    printf("|");
    fflush(stdout);

    reset_counter();
}

void reset_counter() {
    random_table_idx = 0;
}

packed masked_sbox_first_order(packed inp_shares, const uint32_t* s_box) {
    uint32_t t[4] = { 0 };
    uint4_t r = get_rand();
    uint4_t r_prime = uint4_add(r, uint4_neg(SHARE_0(inp_shares)));
    uint4_t off_left = 5 * (r_prime % 4), off_right = 20 - off_left;

    packed output_shares = get_rand();
    uint4_t neg_out_summed_4 = uint4_neg(SHARE_0(output_shares));
    uint32_t neg_out_summed_32 = (neg_out_summed_4 << 15) | (neg_out_summed_4 << 10) | (neg_out_summed_4 << 5) | neg_out_summed_4;
    uint4_t inp_1 = SHARE_1(inp_shares);
    
    for (uint4_t a = 0, curr = r_prime / 4; a < 0x10; a += 4, curr = (curr + 1) % 4) {
        uint32_t s = (s_box[uint4_add(inp_1, a) / 4] + neg_out_summed_32);
        t[curr] |= (s >> off_left);
        t[(curr + 1) % 4] |= (s << off_right) & 0b01111011110111101111;
    }

    uint4_t idx = uint4_add(r, inp_1 % 4);
    return (output_shares & MASK_0) | uint4_new(t[idx / 4] >> (5 * (3 - (idx % 4))));
}

/*packed masked_sbox_second_order(packed inp_shares, const uint32_t* s_box) {
    uint32_t t[4] = { 0 };
    uint4_t r = get_rand();
    uint4_t r_prime = uint4_add(r, uint4_neg(uint4_add(SHARE_0(inp_shares), SHARE_1(inp_shares))));
    uint4_t off_left = 5 * (r_prime % 4), off_right = 20 - off_left;

    packed output_shares = get_rand();
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
}*/
