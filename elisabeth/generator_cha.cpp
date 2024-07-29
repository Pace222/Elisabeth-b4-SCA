#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "generator_cha.h"

#define CHACHA_BATCH_SIZE 64 * (1 + 5 * KEYROUND_WIDTH_4 / 64) /* Number of elements to precompute at a time */

/**
 * \brief          Generate a new batch of random numbers
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     batch: Output variable used to store the new batch
 * \param[in]      r_cha: The PRNG context
 */
void new_batch_cha(uint8_t* batch, rng_cha* r_cha) {
    ECRYPT_keystream_bytes(&r_cha->ctx, batch, CHACHA_BATCH_SIZE);
}

/**
 * \brief          Return a uniform number with ChaCha20, sampling from the current batch, recomputing the latter if necessary
 * \param[in]      r: The PRNG context
 * \param[in]      batch: The batch of random numbers
 * \return         A uniform number
 */
uint8_t random_uniform_cha(rng* r, uint8_t* batch) {
    rng_cha* r_cha = (rng_cha*) r;

    if (r_cha->batch_idx >= CHACHA_BATCH_SIZE) {
        /* New batch */
        new_batch_cha(batch, r_cha);
        r_cha->batch_idx = 0;
    }

    return batch[r_cha->batch_idx++];
}

/**
 * \brief          Copy one ChaCha20 PRNG context to another
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     dest: The destination PRNG context
 * \param[in]      src: The source PRNG context
 */
void copy_cha(rng* dest, const rng* src) {
    rng_cha* dest_cha = (rng_cha*) dest;
    rng_cha* src_cha = (rng_cha*) src;

    /* Copy mother context*/
    copy_rng(&dest_cha->r, &src_cha->r);
    
    /* Copy ChaCha20-specific context */
    memcpy(&dest_cha->ctx, &src_cha->ctx, sizeof(src_cha->ctx));
    dest_cha->batch_idx = src_cha->batch_idx;
}

/**
 * \brief          Restore the context of a previous ChaCha20 PRNG context
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     r: Output variable used to store the updated PRNG context
 */
void next_elem_cha(rng* r) {
    rng_cha* r_cha = (rng_cha*) r;

    uint8_t batch[CHACHA_BATCH_SIZE];
    new_batch_cha(batch, r_cha);
    r_cha->batch_idx = 0;

    precompute_prng(&r_cha->r, batch);
}

/**
 * \brief          Initialize a new ChaCha20 PRNG context
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     r_cha: Output variable used to store the PRNG context
 * \param[in]      seed_little_end: Seed (IV) in little-endian
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 */
void rng_new_cha(rng_cha* r_cha, const uint8_t* seed_little_end, int mode) {

    rng_new(&r_cha->r, mode, random_uniform_cha, copy_cha, next_elem_cha);
    
    ECRYPT_init_ctx(&r_cha->ctx, seed_little_end);

    uint8_t batch[CHACHA_BATCH_SIZE];
    new_batch_cha(batch, r_cha);
    r_cha->batch_idx = 0;

    precompute_prng(&r_cha->r, batch);
}
