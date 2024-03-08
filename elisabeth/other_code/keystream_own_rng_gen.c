#include "keystream.h"


void random_whitened_subset(uint4_t* keyround, uint4_t* key, rng* r) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int KEY_WIDTH = r->mode ? KEY_WIDTH_4 : KEY_WIDTH_B4;
    size_t* indices = r->mode ? r->indices._4 : r->indices._b4;

    // Select a random subset without repetition of size KEYROUND_WIDTH from the key
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        size_t j = gen_range(r, i, KEY_WIDTH);
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    // Generate a random whitening mask
    uint4_t whitening[KEYROUND_WIDTH];
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        whitening[i] = uint4_new(random_uniform(r));
    }

    // Apply the whitening mask to the selected subset
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        keyround[i] = uint4_add(key[indices[i]], whitening[i]);
    }
}

uint4_t filter(uint4_t* keyround, int mode) {
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