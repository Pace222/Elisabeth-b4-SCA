#include "masking.h"
#include <stdio.h>

#define RANDOM_TABLE_SIZE 2048
masked RANDOM_TABLE[RANDOM_TABLE_SIZE];
size_t random_table_idx = 0;

/**
 * \brief          Generate the missing share for the target value
 * \param[in]      rand_shares: The already generated share
 * \param[in]      target_value: The target value
 * \return         The two shares that sum up to the target value
 * \hideinitializer
 */
#define shares_from_value(rand_shares, target_value) (((rand_shares) & MASK_0) | (((target_value) - SHARE_0((rand_shares))) & MASK_1))

/**
 * \brief          Returns a random number (RSI or mask share)
 * \return         The keystream
 */
uint32_t get_rand() {
    return RANDOM_TABLE[random_table_idx++];
}

/**
 * \brief          Get back the represented number: combination of the shares
 * \param[in]      shares: The shares
 * \return         The unmasked value
 */
uint4_t consume_shares(masked shares) {
    return uint4_add(SHARE_0(shares), SHARE_1(shares));
}

/**
 * \brief          Print shares
 * \param[in]      s: The shares
 * \return         The shares
 */
uint32_t printed_shares(masked s) {
    printf("%X", SHARE_0(s));
    printf("%X", SHARE_1(s));
    return s;
}

/**
 * \brief          Print a random number
 * \param[in]      r: The random number
 * \return         The random number
 */
uint32_t printed_rand(uint32_t r) {
    printf("%02X", r);
    return r;
}

/**
 * \brief          Generate mask shares for all given values 
 * \note           This function does not return a value, it stores it to a pointer instead 
 * \param[out]     masked_values: Output variable used to store the masked values
 * \param[in]      values: The values to mask
 * \param[in]      length: The number of values
 */
void init_shares(masked* masked_values, uint4_t* values, size_t length) {
    for (size_t i = 0; i < length; i++) {
        masked shares = rand();
        masked_values[i] = printed_shares(shares_from_value(shares, values[i]));
    }
    printf("|");
    fflush(stdout);
}

/**
 * \brief          Generate null mask shares for all given values 
 * \note           This function does not return a value, it stores it to a pointer instead 
 * \param[out]     masked_values: Output variable used to store the null-masked values
 * \param[in]      values: The values to mask
 * \param[in]      length: The number of values
 */
void init_null_shares(masked* masked_values, uint4_t* values, size_t length) {
    for (size_t i = 0; i < length; i++) {
        masked_values[i] = printed_shares((masked) values[i]);
    }
    printf("|");
    fflush(stdout);
}

/**
 * \brief          Generate the appropriate number of random mask shares needed internally in the algorithm
 */
void generate_masking_random_table() {
    uint32_t r = rand();
    RANDOM_TABLE[0] = shares_from_value(r, 0);
    reset_masking_counter();
}

/**
 * \brief          Generate the appropriate number of null mask shares needed internally in the algorithm
 */
void generate_null_masking_random_table() {
    RANDOM_TABLE[0] = 0;
    reset_masking_counter();
}

/**
 * \brief          Generate the appropriate number of random start indices (RSIs) needed internally in the algorithm
 */
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

    reset_masking_counter();
}

/**
 * \brief          Generate the appropriate number of random mask shares and random start indices (RSIs) needed internally in the algorithm
 */
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

    reset_masking_counter();
}

/**
 * \brief          Reset the head
 */
void reset_masking_counter() {
    random_table_idx = 0;
}

/**
 * \brief          Implementation of the 2-share arithmetic masked S-Box evaluation secure against first-order
 * \param[in]      inp_shares: The masked S-Box input
 * \param[in]      s_box: The S-Box to evaluate
 * \return         The masked output of the S-Box
 */
masked masked_sbox_first_order(masked inp_shares, const uint4_t s_box[16][16]) {
    masked inp_mask_0 = inp_shares & MASK_0;
    masked inp_mask_1 = inp_shares & MASK_1;

    return inp_mask_0 | s_box[inp_mask_0 >> 8][inp_mask_1];
}
