#include "delays.h"
#include <string.h>

#include <SHA512.h>
#define HASH_SIZE SHA512::HASH_SIZE / 2
SHA512 H;

uint16_t bias[N_OPERATIONS];
uint16_t hash_chain[HASH_SIZE];
uint4_t prev_key[KEYROUND_WIDTH_B4];

void hash(void* dest, void* src, size_t src_len) {
    H.reset();
    H.update(src, src_len);
    H.finalize(dest, 2 * HASH_SIZE);
}

void compute_delays(int compute_bias) {
    for (int i = 0; i < N_OPERATIONS; i += HASH_SIZE) {
        for (int j = 0; j < HASH_SIZE; ++j) {
            if (i + j >= N_OPERATIONS) {
                break;
            }
            if (compute_bias) {
                bias[i + j] = hash_chain[j] & 0b111111000;
            }
            delays[i + j] = bias[i + j] | (hash_chain[j] & 0b111);
        }
        hash(hash_chain, hash_chain, sizeof(hash_chain));
    }
}

void init_chain(uint8_t* device_secret, size_t secret_len) {
    hash(hash_chain, device_secret, secret_len);
    compute_delays(1);
    memset(prev_key, 0, sizeof(prev_key));

    head = 0;
}

void new_encryption(uint4_t* new_key) {
    if (memcmp(prev_key, new_key, sizeof(prev_key))) {
        compute_delays(0);
    }

    head = 0;
}
