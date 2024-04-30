#include "keystream.h"

void random_whitened_subset(uint4_t* keyround, const uint4_t* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    // Apply the whitening mask to the selected subset
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        keyround[i] = uint4_add(key[r->indices[i]], r->whitening[i]);
    }
}

void masked_random_whitened_subset(packed* keyround, const packed* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    // Apply the whitening mask to the selected subset
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        keyround[i] = masked_addition_constant(key[r->indices[i]], r->whitening[i]);
    }
}

void shuffled_random_whitened_subset(uint4_t* keyround, const uint4_t* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    // Apply the whitening mask to the selected subset
    int loop_bound = KEYROUND_WIDTH;
    int start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        int final_index = start_index + i;
        int cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;
        
        keyround[final_index] = uint4_add(key[r->indices[final_index]], r->whitening[final_index]);
    }
}

void masked_shuffled_random_whitened_subset(packed* keyround, const packed* key, const rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    // Apply the whitening mask to the selected subset
    int loop_bound = KEYROUND_WIDTH;
    int start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        int final_index = start_index + i;
        int cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        keyround[final_index] = masked_addition_constant(key[r->indices[final_index]], r->whitening[final_index]);
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

packed masked_filter(const packed* keyround_shares, int mode) {
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    packed (*masked_filter_block) (const packed*) = mode ? masked_filter_block_4 : masked_filter_block_b4;

    // Split the keyround into blocks of size BLOCK_WIDTH and apply function filter_block for each block
    packed res_key_shares = get_rand();                                                            // uint4_t res_key = uint4_new(0);
    for (int i = 0; i < KEYROUND_WIDTH; i += BLOCK_WIDTH) {
        res_key_shares = masked_addition(res_key_shares, masked_filter_block(keyround_shares + i)); // res_key = uint4_add(res_key, filter_block(block));
    }
    return res_key_shares;
}

uint4_t shuffled_filter(const uint4_t* keyround, int mode) {
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    uint4_t (*shuffled_filter_block) (const uint4_t*) = mode ? shuffled_filter_block_4 : shuffled_filter_block_b4;

    // Split the keyround into blocks of size BLOCK_WIDTH and apply function filter_block for each block
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

packed masked_shuffled_filter(const packed* keyround_shares, int mode) {
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    packed (*masked_shuffled_filter_block) (const packed*) = mode ? masked_shuffled_filter_block_4 : masked_shuffled_filter_block_b4;

    // Split the keyround into blocks of size BLOCK_WIDTH and apply function filter_block for each block
    packed res_key_shares = get_rand();                                                            // uint4_t res_key = uint4_new(0);
    int loop_bound = KEYROUND_WIDTH / BLOCK_WIDTH;
    int start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        int final_index = start_index + i;
        int cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        const packed* block = keyround_shares + BLOCK_WIDTH * final_index;
        res_key_shares = masked_addition(res_key_shares, masked_shuffled_filter_block(block));               // res_key = uint4_add(res_key, filter_block(block));
    }
    return res_key_shares;
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