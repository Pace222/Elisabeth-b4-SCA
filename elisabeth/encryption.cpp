#include <stdlib.h>

#include "encryption.h"

void encrypt(uint4_t* ciphertext, const uint4_t* plaintext, const uint4_t* key, const rng** r, int length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {
        uint4_t keyround[KEYROUND_WIDTH];
        random_whitened_subset(keyround, key, r[i]);
        uint4_t filtered_key = filter(keyround, r[i]->mode);
        ciphertext[i] = uint4_add(plaintext[i], filtered_key);
    }
}

void masked_encrypt(packed* ciphertext, const packed* plaintext, const packed* key, const rng** r, int length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {
        packed keyround[KEYROUND_WIDTH];
        masked_random_whitened_subset(keyround, key, r[i]);
        packed filtered_key = masked_filter(keyround, r[i]->mode);
        ciphertext[i] = masked_addition(plaintext[i], filtered_key);
    }
}

void decrypt(uint4_t* decrypted, const uint4_t* ciphertext, const uint4_t* key, const rng** r, int length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {
        uint4_t keyround[KEYROUND_WIDTH];
        random_whitened_subset(keyround, key, r[i]);
        uint4_t filtered_key = filter(keyround, r[i]->mode);
        decrypted[i] = uint4_add(ciphertext[i], uint4_neg(filtered_key));
    }
}

void masked_decrypt(packed* decrypted, const packed* ciphertext, const packed* key, const rng** r, int length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {
        packed keyround[KEYROUND_WIDTH];
        masked_random_whitened_subset(keyround, key, r[i]);
        packed filtered_key = masked_filter(keyround, r[i]->mode);
        decrypted[i] = masked_addition(ciphertext[i], masked_negation(filtered_key));
    }
}
