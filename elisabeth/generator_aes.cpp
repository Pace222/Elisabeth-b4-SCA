#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "generator_aes.h"

#define AES_BATCH_SIZE AES_BLOCKLEN * (1 + 5 * KEYROUND_WIDTH_4 / AES_BLOCKLEN)    /* Number of elements to precompute at a time */

/**
 * \brief          Add a value to a little-endian array
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     array: The target array representing a number in little-endian order
 * \param[in]      length: The length of the target array
 * \param[in]      carry: The value to add
 */
void add_sub(uint8_t* array, size_t length, int carry) {
    for (int i = 0; i < length; i++) {
        int sum = array[i] + carry;
        array[i] = sum & 0xFF;
        carry = sum >> 8;
        if (carry == 0) break;
    }
}

/**
 * \brief          Generate a new batch of random numbers and update the internal counter
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     batch: Output variable used to store the batch of random numbers
 * \param[in]      r_aes: The PRNG context
 */
void inc_and_new_batch_aes(uint8_t* batch, rng_aes* r_aes) {
    for (uint8_t* b = batch; b < batch + AES_BATCH_SIZE; b += AES_BLOCKLEN) {
        memcpy(b, r_aes->ctr, AES_BLOCKLEN);
        AES_ECB_encrypt(&r_aes->ctx, b);
        add_sub(r_aes->ctr, AES_BLOCKLEN, 1);
    }
}

/**
 * \brief          Recompute the previous batch
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     batch: Output variable used to store the batch of random numbers
 * \param[in]      r_aes: The PRNG context
 */
void recompute_last_batch_aes(uint8_t* batch, rng_aes* r_aes) {
    add_sub(r_aes->ctr, AES_BLOCKLEN, -AES_BATCH_SIZE/AES_BLOCKLEN);
    inc_and_new_batch_aes(batch, r_aes);
}

/**
 * \brief          Return a uniform number with AES, sampling from the current batch, recomputing the latter if necessary
 * \param[in]      r: The PRNG context
 * \param[in]      batch: The batch of random numbers
 * \return         A uniform number
 */
uint8_t random_uniform_aes(rng* r, uint8_t* batch) {
    rng_aes* r_aes = (rng_aes*) r;

    if (r_aes->batch_idx >= AES_BATCH_SIZE) {
        /* New batch */
        inc_and_new_batch_aes(batch, r_aes);
        r_aes->batch_idx = 0;
    }

    return batch[r_aes->batch_idx++];
}

/**
 * \brief          Copy one AES PRNG context to another
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     dest: The destination PRNG context
 * \param[in]      src: The source PRNG context
 */
void copy_aes(rng* dest, const rng* src) {
    rng_aes* dest_aes = (rng_aes*) dest;
    rng_aes* src_aes = (rng_aes*) src;

    /* Copy mother context*/
    copy_rng(&dest_aes->r, &src_aes->r);
    
    /* Copy AES-specific context */
    memcpy(&dest_aes->ctx, &src_aes->ctx, sizeof(src_aes->ctx));
    memcpy(dest_aes->ctr, src_aes->ctr, AES_BLOCKLEN);
    dest_aes->batch_idx = src_aes->batch_idx;
}

/**
 * \brief          Restore the context of a previous AES PRNG context
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     r: Output variable used to store the updated PRNG context
 */
void next_elem_aes(rng* r) {
    rng_aes* r_aes = (rng_aes*) r;

    uint8_t batch[AES_BATCH_SIZE];
    recompute_last_batch_aes(batch, r_aes);

    precompute_prng(&r_aes->r, batch);
}

/**
 * \brief          Initialize a new AES PRNG context
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     r_aes: Output variable used to store the PRNG context
 * \param[in]      seed_little_end: Seed (IV) in little-endian
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 */
void rng_new_aes(rng_aes* r_aes, const uint8_t* seed_little_end, int mode) {

    rng_new(&r_aes->r, mode, random_uniform_aes, copy_aes, next_elem_aes);
    
    AES_init_ctx(&r_aes->ctx, seed_little_end);
    /* Start counter at 0 */
    memset(r_aes->ctr, 0, AES_BLOCKLEN);
    r_aes->batch_idx = 0;

    uint8_t batch[AES_BATCH_SIZE];
    inc_and_new_batch_aes(batch, r_aes);

    precompute_prng(&r_aes->r, batch);
}
