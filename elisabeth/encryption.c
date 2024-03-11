#include <stdlib.h>

#include "encryption.h"

void encrypt(uint4_t* ciphertext, const uint4_t* plaintext, const uint4_t* key, const rng** r, size_t length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {
        uint4_t keyround[KEYROUND_WIDTH];
        random_whitened_subset(keyround, key, r[i]);
        uint4_t filtered_key = filter(keyround, r[i]->mode);
        ciphertext[i] = uint4_add(plaintext[i], filtered_key);
    }
}

void decrypt(uint4_t* decrypted, const uint4_t* ciphertext, const uint4_t* key, const rng** r, size_t length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {
        uint4_t keyround[KEYROUND_WIDTH];
        random_whitened_subset(keyround, key, r[i]);
        uint4_t filtered_key = filter(keyround, r[i]->mode);
        decrypted[i] = uint4_add(ciphertext[i], uint4_neg(filtered_key));
    }
}