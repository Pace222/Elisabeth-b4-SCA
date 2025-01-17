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

/**
 * \brief          Hash the source buffer into the destination
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     dest: The destination buffer
 * \param[in]      src: The source buffer
 * \param[in]      src_len: The length of the source buffer
 */
void hash(void* dest, void* src, size_t src_len) {
    H.reset();
    H.update(src, src_len);
    H.finalize(dest, HASH_SIZE);
}

/**
 * \brief          For every operation, computes its associated delay as described by our countermeasure
 */
void compute_delays() {
    for (int i = 0; i < N_OPERATIONS; i += 2 * HASH_SIZE) {
        /* Rehash until we have enough entropy for every operation */
        hash(&hash_chain_and_prev_key.hash_chain, &hash_chain_and_prev_key, sizeof(hash_chain_and_prev_key));
        for (int j = 0; j < HASH_SIZE; ++j) {
            /* Divide each byte into two nibbles, one per operation */
            if (i + 2*j >= N_OPERATIONS)
                break;
            delays[i + 2*j]     = bias | ((hash_chain_and_prev_key.hash_chain[j] & 0b11110000) >> 4);

            if (i + 2*j + 1 >= N_OPERATIONS)
                break;
            delays[i + 2*j + 1] = bias | ((hash_chain_and_prev_key.hash_chain[j] & 0b00001111) >> 0);
        }
    }
}

/**
 * \brief          Initialize the bias according to the device secret and comptes the delays for every operation
 * \param[in]      device_secret: The device secret
 * \param[in]      sercret_len: The length of the device secret
 */
void init_chain(uint8_t* device_secret, size_t secret_len) {
    hash(&hash_chain_and_prev_key.hash_chain, device_secret, secret_len);
    bias = hash_chain_and_prev_key.hash_chain[0] & 0b11110000;

    memset(&hash_chain_and_prev_key.prev_key, 0, sizeof(hash_chain_and_prev_key.prev_key));    /* Initial key set to 0 */
    compute_delays();

    delay_head = 0;
}

/**
 * \brief          When a new encryption is started, one checks if the key has changed, updating the delays if it is the case
 * \param[in]      new_key: The key used in the next encryption
 */
void new_encryption(uint4_t* new_key) {
    if (memcmp(hash_chain_and_prev_key.prev_key, new_key, sizeof(hash_chain_and_prev_key.prev_key))) {
        memcpy(hash_chain_and_prev_key.prev_key, new_key, sizeof(hash_chain_and_prev_key.prev_key));
        compute_delays();
    }

    delay_head = 0;
}

/**
 * \brief          Print the delays of every operation
 */
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