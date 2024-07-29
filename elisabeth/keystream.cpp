#include "keystream.h"
#include "delays.h"

/**
 * \brief          RWS
 * \note           This function does not return a value, it stores it to a pointer instead 
 * \param[out]     keyround: Output variable used to store the keyround
 * \param[in]      key: The key
 * \param[in]      r: The PRNG context
 */
void random_whitened_subset(uint4_t* keyround, const uint4_t* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    /* Apply the whitening mask to the selected subset of key elements */
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        keyround[i] = uint4_add(key[r->indices[i]], r->whitening[i]);
    }
}

/**
 * \brief          2-share arithmetic masked RWS
 * \note           This function does not return a value, it stores it to a pointer instead 
 * \param[out]     keyround: Output variable used to store the masked keyround
 * \param[in]      key: The masked key
 * \param[in]      r: The PRNG context
 */
void masked_random_whitened_subset(masked* keyround, const masked* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    /* Apply the whitening mask to the selected subset of key elements */
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        keyround[i] = masked_addition_constant(key[r->indices[i]], r->whitening[i]);
    }
}

/**
 * \brief          Shuffled RWS
 * \note           This function does not return a value, it stores it to a pointer instead 
 * \param[out]     keyround: Output variable used to store the keyround
 * \param[in]      key: The key
 * \param[in]      r: The PRNG context
 */
void shuffled_random_whitened_subset(uint4_t* keyround, const uint4_t* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    /* Apply the whitening mask to the selected subset of key elements */
    int loop_bound = KEYROUND_WIDTH;
    int start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        int final_index = start_index + i;
        int cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;
        
        keyround[final_index] = uint4_add(key[r->indices[final_index]], r->whitening[final_index]);
    }
}

/**
 * \brief          2-share arithmetic masked and shuffled RWS
 * \note           This function does not return a value, it stores it to a pointer instead 
 * \param[out]     keyround: Output variable used to store the masked keyround
 * \param[in]      key: The masked key
 * \param[in]      r: The PRNG context
 */
void masked_shuffled_random_whitened_subset(masked* keyround, const masked* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    /* Apply the whitening mask to the selected subset of key elements */
    int loop_bound = KEYROUND_WIDTH;
    int start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        int final_index = start_index + i;
        int cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        keyround[final_index] = masked_addition_constant(key[r->indices[final_index]], r->whitening[final_index]);
    }
}

/**
 * \brief          Delayed RWS as part of our novel countermeasure
 * \note           This function does not return a value, it stores it to a pointer instead 
 * \param[out]     keyround: Output variable used to store the keyround
 * \param[in]      key: The key
 * \param[in]      r: The PRNG context
 */
void random_whitened_subset_delayed(uint4_t* keyround, const uint4_t* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    /* Apply the whitening mask to the selected subset of key elements */
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        delay_operation();
        keyround[i] = uint4_add(key[r->indices[i]], r->whitening[i]);
    }
}

/**
 * \brief          Execution of 14 rounds sequentially
 * \param[in]      keyround: The keyround
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 * \return         The keystream
 */
uint4_t filter(const uint4_t* keyround, int mode) {
    /* Set the appropriate variables according to the mode */
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    uint4_t (*filter_block) (const uint4_t*) = mode ? filter_block_4 : filter_block_b4;

    /* Split the keyround into blocks and apply the filtering function for each block */
    uint4_t res_key = uint4_new(0);
    for (const uint4_t* block = keyround; block < keyround + KEYROUND_WIDTH; block += BLOCK_WIDTH) {
        res_key = uint4_add(res_key, filter_block(block));
    }
    return res_key;
}

/**
 * \brief          2-share arithmetic masked execution of 14 rounds sequentially
 * \param[in]      keyround: The masked keyround
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 * \return         The masked keystream
 */
masked masked_filter(const masked* keyround_shares, int mode) {
    /* Set the appropriate variables according to the mode */
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    masked (*masked_filter_block) (const masked*) = mode ? masked_filter_block_4 : masked_filter_block_b4;

    /* Split the keyround into blocks and apply the filtering function for each block */
    masked res_key_shares = get_rand();
    for (int i = 0; i < KEYROUND_WIDTH; i += BLOCK_WIDTH) {
        res_key_shares = masked_addition(res_key_shares, masked_filter_block(keyround_shares + i));
    }
    return res_key_shares;
}

/**
 * \brief          Shuffled execution of 14 rounds sequentially
 * \param[in]      keyround: The keyround
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 * \return         The keystream
 */
uint4_t shuffled_filter(const uint4_t* keyround, int mode) {
    /* Set the appropriate variables according to the mode */
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    uint4_t (*shuffled_filter_block) (const uint4_t*) = mode ? shuffled_filter_block_4 : shuffled_filter_block_b4;

    /* Split the keyround into blocks and apply the filtering function for each block */
    uint4_t res_key = uint4_new(0);
    int loop_bound = KEYROUND_WIDTH / BLOCK_WIDTH;
    int start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        int final_index = start_index + i;
        int cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        const uint4_t* block = keyround + BLOCK_WIDTH * final_index;
        res_key = uint4_add(res_key, shuffled_filter_block(block));
    }
    return res_key;
}

/**
 * \brief          2-share arithmetic masked and shuffled execution of 14 rounds sequentially
 * \param[in]      keyround: The masked keyround
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 * \return         The masked keystream
 */
masked masked_shuffled_filter(const masked* keyround_shares, int mode) {
    /* Set the appropriate variables according to the mode */
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    masked (*masked_shuffled_filter_block) (const masked*) = mode ? masked_shuffled_filter_block_4 : masked_shuffled_filter_block_b4;

    /* Split the keyround into blocks and apply the filtering function for each block */
    masked res_key_shares = get_rand();
    int loop_bound = KEYROUND_WIDTH / BLOCK_WIDTH;
    int start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        int final_index = start_index + i;
        int cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        const masked* block = keyround_shares + BLOCK_WIDTH * final_index;
        res_key_shares = masked_addition(res_key_shares, masked_shuffled_filter_block(block));
    }
    return res_key_shares;
}

/**
 * \brief          Delayed execution of 14 rounds sequentially as part of our novel countermeasure
 * \param[in]      keyround: The keyround
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 * \return         The keystream
 */
uint4_t filter_delayed(const uint4_t* keyround, int mode) {
    /* Set the appropriate variables according to the mode */
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    uint4_t (*filter_block) (const uint4_t*) = mode ? filter_block_4_delayed : filter_block_b4_delayed;

    /* Split the keyround into blocks and apply the filtering function for each block */
    uint4_t res_key = uint4_new(0);
    for (const uint4_t* block = keyround; block < keyround + KEYROUND_WIDTH; block += BLOCK_WIDTH) {
        res_key = uint4_add(res_key, filter_block(block));
    }
    return res_key;
}
