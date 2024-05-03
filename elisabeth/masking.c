#include "masking.h"
#include <stdio.h>

#define RANDOM_TABLE_SIZE 2048
add_packed RANDOM_TABLE[RANDOM_TABLE_SIZE];
size_t random_table_idx = 0;

#define add_shares_from_value(rand_shares, target_value) (((rand_shares) & MASK_0) | (((target_value) - SHARE_0((rand_shares))) & MASK_1))

uint32_t get_rand() {
    return RANDOM_TABLE[random_table_idx++];
}

uint4_t consume_shares(add_packed shares) {
    return uint4_add(SHARE_0(shares), SHARE_1(shares));
}

uint4_t mul_consume_shares(mul_packed shares) {
    return uint4_new(MUL_SHARE_1(shares) / MUL_SHARE_0(shares));
}

uint32_t printed_shares(add_packed s) {
    printf("%X", SHARE_0(s));
    printf("%X", SHARE_1(s));
    return s;
}

uint32_t printed_rand(uint32_t r) {
    printf("%02X", r);
    return r;
}

void init_shares(add_packed* packed_values, uint4_t* values, size_t length) {
    for (size_t i = 0; i < length; i++) {
        add_packed shares = rand();
        packed_values[i] = printed_shares(add_shares_from_value(shares, values[i]));
    }
    printf("|");
    fflush(stdout);
}

void init_null_shares(add_packed* packed_values, uint4_t* values, size_t length) {
    for (size_t i = 0; i < length; i++) {
        packed_values[i] = printed_shares((add_packed) values[i]);
    }
    printf("|");
    fflush(stdout);
}

void generate_masking_random_table() {
    uint32_t r = rand();
    RANDOM_TABLE[0] = add_shares_from_value(r, 0);
    size_t k = 1;
    for (size_t i = 0; i < 14; i++) {
        for (size_t j = 0; j < 18; j++) {
            RANDOM_TABLE[k++] = 1 + rand() % (MUL_MODULO - 1);
            RANDOM_TABLE[k++] = rand() % MUL_MODULO;
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
    RANDOM_TABLE[1] = add_shares_from_value(r, 0);
    RANDOM_TABLE[2] = printed_rand(rand() % 14);
    size_t k = 3;
    for (size_t i = 0; i < 14; i++) {
        // Filter blocks
        RANDOM_TABLE[k++] = printed_rand(rand() % 7);

        RANDOM_TABLE[k++] = printed_rand(rand() % 3);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        for (int j = 0; j < 6; j++) { // First S-box round
            RANDOM_TABLE[k++] = 1 + rand() % (MUL_MODULO - 1);
            RANDOM_TABLE[k++] = rand() % MUL_MODULO;
        }

        RANDOM_TABLE[k++] = printed_rand(rand() % 3);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        for (int j = 0; j < 6; j++) { // Second S-box round
            RANDOM_TABLE[k++] = 1 + rand() % (MUL_MODULO - 1);
            RANDOM_TABLE[k++] = rand() % MUL_MODULO;
        }

        RANDOM_TABLE[k++] = printed_rand(rand() % 2);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        RANDOM_TABLE[k++] = printed_rand(rand() % 6);
        for (int j = 0; j < 6; j++) { // Third S-box round
            RANDOM_TABLE[k++] = 1 + uint4_new(rand());
            RANDOM_TABLE[k++] = rand() % MUL_MODULO;
        }
    }
    printf("|");
    fflush(stdout);

    reset_counter();
}

void reset_counter() {
    random_table_idx = 0;
}

mul_packed add_to_mul_shares(add_packed additive_shares) {
    uint8_t mul_share_0 = get_rand();
    uint8_t multiplied_0 = (SHARE_0(additive_shares) * mul_share_0) % MUL_MODULO;
    uint8_t multiplied_1 = (SHARE_1(additive_shares) * mul_share_0) % MUL_MODULO;
    uint8_t mul_share_1 = (multiplied_0 + multiplied_1) % MUL_MODULO;

    return (((uint32_t) mul_share_0) << 28) | mul_share_1;
}

add_packed mul_to_add_shares(mul_packed multiplicative_shares) {
    uint8_t add_share_0 = get_rand();

    uint4_t mul_share_0 = MUL_SHARE_0(multiplicative_shares);
    uint8_t mul_share_1 = MUL_SHARE_1(multiplicative_shares);

    uint8_t inv_mul_share_0;
    for (uint8_t guess = 1; guess < MUL_MODULO; guess++) {
        if ((guess * mul_share_0) % MUL_MODULO == 1) {
            inv_mul_share_0 = guess;
        }
    }
    uint4_t add_share_1 = ((mul_share_1 - add_share_0 * mul_share_0) * inv_mul_share_0) % MUL_MODULO;
    // TODO: add_share_0 and add_share_1 are additive shares modulo 17. How to turn them into shares of modulo 16 ?? 
    return 0;// (((uint32_t) add_share_0) << 16) | add_share_1;
}

add_packed masked_sbox_first_order(add_packed inp_add_shares, const uint32_t* s_box) {
    mul_packed inp_mul_shares = add_to_mul_shares(inp_add_shares);
    uint8_t mul_share_1 = MUL_SHARE_1(inp_mul_shares);
    mul_packed mul_output = s_box[mul_share_1];
    
    return mul_to_add_shares(mul_output);
}
