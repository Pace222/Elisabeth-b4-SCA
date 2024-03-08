#include "keystream.h"


void random_whitened_subset(uint4_t* keyround, const uint4_t* key, rng* r, const uint32_t* precomputed_random_values) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int KEY_WIDTH = r->mode ? KEY_WIDTH_4 : KEY_WIDTH_B4;

    // Select a random subset without repetition of size KEYROUND_WIDTH from the key
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        uint32_t j = precomputed_random_values[i];
        size_t tmp = r->indices[i];
        r->indices[i] = r->indices[j];
        r->indices[j] = tmp;
    }

    // Apply the whitening mask to the selected subset
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        keyround[i] = uint4_add(key[r->indices[i]], precomputed_random_values[KEYROUND_WIDTH + i]);
    }
}

uint4_t filter(const uint4_t* keyround, int mode) {
    int KEYROUND_WIDTH = mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int BLOCK_WIDTH = mode ? BLOCK_WIDTH_4 : BLOCK_WIDTH_B4;
    uint4_t (*filter_block) (uint4_t*) = mode ? filter_block_4 : filter_block_b4;

    // Split the keyround into blocks of size BLOCK_WIDTH and apply function filter_block for each block
    uint4_t res_key = uint4_new(0);
    for (uint4_t* block = keyround; block < keyround + KEYROUND_WIDTH; block += BLOCK_WIDTH) {
        res_key = uint4_add(res_key, filter_block(block));
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