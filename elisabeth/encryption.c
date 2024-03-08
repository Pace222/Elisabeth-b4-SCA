#include "encryption.h"

void encrypt(uint4_t* ciphertext, uint4_t* plaintext, uint4_t* key, rng* r, size_t length, uint32_t* precomputed_random_values) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    uint4_t filtered_key;
    for (int i = 0; i < length; i++) {
        uint4_t keyround[KEYROUND_WIDTH];
        random_whitened_subset(keyround, key, r, precomputed_random_values + 2 * KEYROUND_WIDTH * i);
        filtered_key = filter(keyround, r->mode);
        ciphertext[i] = uint4_add(plaintext[i], filtered_key);
    }
}

void decrypt(uint4_t* decrypted, uint4_t* ciphertext, uint4_t* key, rng* r, size_t length, uint32_t* precomputed_random_values) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    for (int i = 0; i < length; i++) {
        uint4_t keyround[KEYROUND_WIDTH];
        random_whitened_subset(keyround, key, r, precomputed_random_values + 2 * KEYROUND_WIDTH * i);
        uint4_t filtered_key = filter(keyround, r->mode);
        decrypted[i] = uint4_add(ciphertext[i], uint4_neg(filtered_key));
    }
}