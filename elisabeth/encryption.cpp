#include <stdlib.h>

#include "encryption.h"

/**
 * \brief          Encryption of a plaintext message
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     ciphertext: Output variable used to store the ciphertext message 
 * \param[in]      plaintext: The plaintext message
 * \param[in]      key: The key
 * \param[in]      r: The instances of the PRNGs used in the RWS, one for each element to encrypt
 * \param[in]      length: The number of elements in the message
 */
void encrypt(uint4_t* ciphertext, const uint4_t* plaintext, const uint4_t* key, const rng** r, int length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {         /* For every element, */
        uint4_t keyround[KEYROUND_WIDTH];
        random_whitened_subset(keyround, key, r[i]);   /* RWS */
        uint4_t filtered_key = filter(keyround, r[i]->mode);   /* 14 rounds */
        ciphertext[i] = uint4_add(plaintext[i], filtered_key); /* Final addition*/
    }
}

/**
 * \brief          2-share arithmetic masked encryption of a plaintext message
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     ciphertext: Output variable used to store the masked ciphertext message 
 * \param[in]      plaintext: The masked plaintext message
 * \param[in]      key: The masked key
 * \param[in]      r: The instances of the PRNGs used in the RWS, one for each element to encrypt
 * \param[in]      length: The number of elements in the message
 */
void masked_encrypt(masked* ciphertext, const masked* plaintext, const masked* key, const rng** r, int length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {         /* For every element, */
        masked keyround[KEYROUND_WIDTH];
        masked_random_whitened_subset(keyround, key, r[i]);    /* Masked RWS */
        masked filtered_key = masked_filter(keyround, r[i]->mode); /* Masked 14 rounds*/
        ciphertext[i] = masked_addition(plaintext[i], filtered_key);   /* Masked final addition*/
    }
}

/**
 * \brief          Decryption of a ciphertext message
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     decrypted: Output variable used to store the decrypted ciphertext message
 * \param[in]      ciphertext: The ciphertext message
 * \param[in]      key: The key
 * \param[in]      r: The instances of the PRNGs used in the RWS, one for each element to decrypt
 * \param[in]      length: The number of elements in the message
 */
void decrypt(uint4_t* decrypted, const uint4_t* ciphertext, const uint4_t* key, const rng** r, int length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {         /* For every element, */
        uint4_t keyround[KEYROUND_WIDTH];
        random_whitened_subset(keyround, key, r[i]);   /* RWS */
        uint4_t filtered_key = filter(keyround, r[i]->mode);   /* 14 rounds */
        decrypted[i] = uint4_add(ciphertext[i], uint4_neg(filtered_key));  /* Final subtraction */
    }
}

/**
 * \brief          2-share arithmetic masked decryption of a ciphertext message
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     decrypted: Output variable used to store the masked decrypted ciphertext message
 * \param[in]      ciphertext: The masked ciphertext message
 * \param[in]      key: The masked key
 * \param[in]      r: The instances of the PRNGs used in the RWS, one for each element to decrypt
 * \param[in]      length: The number of elements in the message
 */
void masked_decrypt(masked* decrypted, const masked* ciphertext, const masked* key, const rng** r, int length) {
    int KEYROUND_WIDTH = r[0]->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;

    for (int i = 0; i < length; i++) {         /* For every element, */
        masked keyround[KEYROUND_WIDTH];
        masked_random_whitened_subset(keyround, key, r[i]);    /* Masked RWS */
        masked filtered_key = masked_filter(keyround, r[i]->mode); /* Masked 14 rounds*/
        decrypted[i] = masked_addition(ciphertext[i], masked_negation(filtered_key));  /* Masked final subtraction*/
    }
}
