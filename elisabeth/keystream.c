#include "keystream.h"

void random_whitened_subset(uint4_t* keyround, const uint4_t* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    // Apply the whitening mask to the selected subset
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        keyround[i] = uint4_add(key[r->indices[i]], r->whitening[i]);
    }
}

void protected_random_whitened_subset(uint4_t keyround[][N_SHARES], const uint4_t key[][N_SHARES], const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    // Apply the whitening mask to the selected subset
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        masked_addition_constant(keyround[i], key[r->indices[i]], r->whitening[i]);
    }
}

uint4_t filter(const uint4_t* keyround, int mode) {
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    uint4_t (*filter_block) (const uint4_t*) = mode ? filter_block_4 : filter_block_b4;

    // Split the keyround into blocks of size BLOCK_WIDTH and apply function filter_block for each block
    uint4_t res_key = uint4_new(0);
    for (const uint4_t* block = keyround; block < keyround + KEYROUND_WIDTH; block += BLOCK_WIDTH) {
        res_key = uint4_add(res_key, filter_block(block));
    }
    return res_key;
}

void protected_filter(uint4_t res_key_shares[N_SHARES], const uint4_t keyround_shares[][N_SHARES], int mode) {
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    void (*protected_filter_block) (uint4_t*, const uint4_t[][N_SHARES]) = mode ? protected_filter_block_4_mask_everything : protected_filter_block_b4_mask_everything;

    // Split the keyround into blocks of size BLOCK_WIDTH and apply function filter_block for each block
    init_shares(res_key_shares, uint4_new(0));                                                    // uint4_t res_key = uint4_new(0);
    for (int i = 0; i < KEYROUND_WIDTH_B4; i += BLOCK_WIDTH_B4) {
        uint4_t res_shares[N_SHARES];                                                             // res_key = uint4_add(res_key, filter_block(block));
        protected_filter_block(res_shares, keyround_shares + i);                                  // res_key = uint4_add(res_key, filter_block(block));
        masked_addition(res_key_shares, res_key_shares, res_shares);                              // res_key = uint4_add(res_key, filter_block(block));
    }
}

/*uint4_t filter_par(uint4_t* keyround) {
    int NUM_BLOCKS = KEYROUND_WIDTH/BLOCK_WIDTH;
    pthread_t threads[NUM_BLOCKS];

    // Create threads to process each block
    for (int i = 0; i < NUM_BLOCKS; i++) {
        pthread_create(&threads[i], 0, filter_block, keyround + i * BLOCK_WIDTH);
    }

    // Join threads
    for (int i = 0; i < NUM_BLOCKS; i++) {
        pthread_join(threads[i], 0);
    }


}*/