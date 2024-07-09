#include "delays.h"
#include <stdio.h>
#include <string.h>

#include "SHA512.h"
#define HASH_SIZE SHA512::HASH_SIZE
SHA512 H;

struct hash_key {
    uint8_t hash_chain[HASH_SIZE];
    uint4_t prev_key[KEY_WIDTH_B4];
};

uint8_t delays[N_OPERATIONS];
size_t delay_head;
uint8_t bias;
struct hash_key hash_chain_and_prev_key;

void hash(void* dest, void* src, size_t src_len) {
    H.reset();
    H.update(src, src_len);
    H.finalize(dest, HASH_SIZE);
}

void compute_delays() {
    for (int i = 0; i < N_OPERATIONS; i += HASH_SIZE) {
        for (int j = 0; j < HASH_SIZE; ++j) {
            if (i + j >= N_OPERATIONS)
                break;

            delays[i + j] = bias | (hash_chain_and_prev_key.hash_chain[j] & 0b111);
        }
        hash(&hash_chain_and_prev_key.hash_chain, &hash_chain_and_prev_key, sizeof(hash_chain_and_prev_key));
    }
}

void init_chain(uint8_t* device_secret, size_t secret_len) {
    hash(&hash_chain_and_prev_key.hash_chain, device_secret, secret_len);
    bias = hash_chain_and_prev_key.hash_chain[0] & 0b11111000;
    memset(&hash_chain_and_prev_key.prev_key, 0, sizeof(hash_chain_and_prev_key.prev_key));
    compute_delays();

    delay_head = 0;
}

void new_encryption(uint4_t* new_key) {
    if (memcmp(hash_chain_and_prev_key.prev_key, new_key, sizeof(hash_chain_and_prev_key.prev_key))) {
        memcpy(hash_chain_and_prev_key.prev_key, new_key, sizeof(hash_chain_and_prev_key.prev_key));
        compute_delays();
    }

    delay_head = 0;
}


void print_delays() {
    printf("(");
    for (int i = 0; i < N_OPERATIONS; ++i) {
        printf("%hX", delays[i]);
        if (i < N_OPERATIONS - 1) {
            printf(";");
        }
    }

    printf(")");
    fflush(stdout);
}