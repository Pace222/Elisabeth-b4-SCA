#include "keystream.h"
#include "masking.h"

void random_whitened_subset(uint4_t* keyround, const uint4_t* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    // Apply the whitening mask to the selected subset
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        keyround[i] = uint4_add(key[r->indices[i]], r->whitening[i]);
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

uint4_t protected_filter_b4_mask_everything(const uint4_t* keyround, int mode) {
    uint4_t res_key_shares[N_SHARES];                                                             // uint4_t res_key = uint4_new(0);
    gen_n_shares(res_key_shares + 1, N_SHARES - 1);                                               // uint4_t res_key = uint4_new(0);
    res_key_shares[0] = uint4_add(0, uint4_neg(uint4_add(res_key_shares[1], res_key_shares[2]))); // uint4_t res_key = uint4_new(0);
    
    // Generate shares for the key
    uint4_t keyround_shares[KEYROUND_WIDTH_B4][N_SHARES];
    for (int i = 0; i < KEYROUND_WIDTH_B4; i++) {
        gen_n_shares(keyround_shares[i] + 1, N_SHARES - 1);
        keyround_shares[i][0] = uint4_add(keyround[i], uint4_neg(uint4_add(keyround_shares[i][1], keyround_shares[i][2])));
    }

    // Split the keyround into blocks of size BLOCK_WIDTH and apply function filter_block for each block
    for (int i = 0; i < KEYROUND_WIDTH_B4; i += BLOCK_WIDTH_B4) {
        uint4_t res_shares[N_SHARES];                                                             // res_key = uint4_add(res_key, filter_block(block));
        protected_filter_block_b4_mask_everything(res_shares, keyround_shares + i);               // res_key = uint4_add(res_key, filter_block(block));
        masked_addition_second_order(res_key_shares, res_key_shares, res_shares);                 // res_key = uint4_add(res_key, filter_block(block));
    }

    return consume_n_shares(res_key_shares, N_SHARES);
}

uint4_t protected_filter_b4_mask_only_input(const uint4_t* keyround, int mode) {    
    // Generate shares for the key
    uint4_t keyround_shares[KEYROUND_WIDTH_B4][N_SHARES];
    for (int i = 0; i < KEYROUND_WIDTH_B4; i++) {
        gen_n_shares(keyround_shares[i] + 1, N_SHARES - 1);
        keyround_shares[i][0] = uint4_add(keyround[i], uint4_neg(uint4_add(keyround_shares[i][1], keyround_shares[i][2])));
    }

    // Split the keyround into blocks of size BLOCK_WIDTH and apply function filter_block for each block
    uint4_t res_key = uint4_new(0);
    for (int i = 0; i < KEYROUND_WIDTH_B4; i += BLOCK_WIDTH_B4) {
        res_key = uint4_add(res_key, protected_filter_block_b4_mask_only_input(keyround_shares + i));
    }

    return res_key;
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