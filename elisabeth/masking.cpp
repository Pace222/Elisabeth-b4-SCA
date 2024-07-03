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

uint32_t printed_shares(packed s) {
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
        packed_values[i] = printed_shares((packed) values[i]);
    }
    printf("|");
    fflush(stdout);
}

void generate_masking_random_table() {
    uint32_t r = rand();
    RANDOM_TABLE[0] = shares_from_value(r, 0);
    reset_counter();
}

void generate_null_masking_random_table() {
    RANDOM_TABLE[0] = 0;
    reset_counter();
}

void generate_shuffling_random_table() {
    RANDOM_TABLE[0] = printed_rand(rand() % 98);
    RANDOM_TABLE[1] = printed_rand(rand() % 14);
    size_t k = 2;
    for (size_t i = 0; i < 14; i++) {
        // Filter blocks
        RANDOM_TABLE[k++] = printed_rand(rand() % 7);

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
        RANDOM_TABLE[k++] = printed_rand(rand() % 7);

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

void reset_counter() {
    random_table_idx = 0;
}

packed masked_sbox_first_order(packed inp_shares, const uint4_t s_box[16][16]) {
    packed inp_mask_0 = inp_shares & MASK_0;
    packed inp_mask_1 = inp_shares & MASK_1;

    return inp_mask_0 | s_box[inp_mask_0 >> 8][inp_mask_1];
}
