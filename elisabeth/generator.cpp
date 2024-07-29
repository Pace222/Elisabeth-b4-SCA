#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "generator.h"

/**
 * \brief          Switch the endianness of the given number
 * \note           This function does not return a value, it stores it to a pointer instead 
 * \param[out]     dest: The destination buffer
 * \param[in]      src: The source buffer
 * \param[in]      length: The length of the source buffer
 */
void switch_endianness(uint8_t* dest, const uint8_t* src, int length) {
    for (int i = 0; i < length; i++) {
        dest[length - 1 - i] = src[i];
    }
}

/**
 * \brief          Generate a random number with n bits
 * \param[in]      r: The PRNG context
 * \param[in]      n: The number of bits
 * \param[in]      batch: The batch of random numbers
 * \return         The random number between 0 and 2^n - 1
 */
uint32_t random_uniform_n_lsb(rng* r, size_t n, uint8_t* batch) {
    uint32_t random_u32 = 0;
    for (int i = 0; i < 4; i++) {
        random_u32 |= (r->gen_rand_uniform(r, batch) << (i * 8));
    }

    return random_u32 >> (32 - n);
}

/**
 * \brief          Generate a random number between the given bounds from the batch
 * \param[in]      r: The PRNG context
 * \param[in]      min: The lower bound
 * \param[in]      max: The higher bound
 * \param[in]      batch: The batch of random numbers
 * \return         The random number between `min` and `max`
 */
uint32_t gen_range(rng* r, size_t min, size_t max, uint8_t* batch) {
    if (min > max - 1) {
        return -1;
    }

    size_t bit_len = (size_t) floor(log2(max - 1 - min)) + 1;
    size_t a = min + random_uniform_n_lsb(r, bit_len, batch);

    while (a >= max) {
        a = min + random_uniform_n_lsb(r, bit_len, batch);
    }
    return a;
}

/**
 * \brief          Precomputes the indices and whitening from the given batch of random numbers
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     r: Output variable used to store the indices and whitening
 * \param[in]      batch: The batch of random numbers
 */
void precompute_prng(rng* r, uint8_t* batch) {
    int KEYROUND_WIDTH = r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4;
    int KEY_WIDTH = r->mode ? KEY_WIDTH_4 : KEY_WIDTH_B4;

    // Select a random subset without repetition of size KEYROUND_WIDTH from the key
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        uint32_t j = gen_range(r, i, KEY_WIDTH, batch);
        size_t tmp = r->indices[i];
        r->indices[i] = r->indices[j];
        r->indices[j] = tmp;
    }
    
    // Generate a random whitening mask
    for (int i = 0; i < KEYROUND_WIDTH; i++) {
        r->whitening[i] = uint4_new(r->gen_rand_uniform(r, batch));
    }
}

/**
 * \brief          Copy one vanilla PRNG context to another
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     dest: The destination PRNG context
 * \param[in]      src: The source PRNG context
 */
void copy_rng(rng* dest, const rng* src) {
    memcpy(dest->indices, src->indices, sizeof(src->indices));
    memcpy(dest->whitening, src->whitening, sizeof(src->whitening));
    dest->mode = src->mode;
    dest->gen_rand_uniform = src->gen_rand_uniform;
    dest->copy = src->copy;
    dest->next_elem = src->next_elem;
}

/**
 * \brief          Initialize a new vanilla PRNG context
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     r: Output variable used to store the PRNG context
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 * \param[in]      gen_rand_uniform: Method that returns a random number
 * \param[in]      copy: Method that copies one PRNG context to another
 * \param[in]      next_elem: Method that restores the context of a previous PRNG context
 */
void rng_new(rng* r, int mode, uint8_t (*gen_rand_uniform)(rng*, uint8_t*), void (*copy)(rng*, const rng*), void (*next_elem)(rng*)) {
    int KEY_WIDTH = mode ? KEY_WIDTH_4 : KEY_WIDTH_B4;

    for (int i = 0; i < KEY_WIDTH; i++) {
        r->indices[i] = i;
    }
    r->mode = mode;
    r->gen_rand_uniform = gen_rand_uniform;
    r->copy = copy;
    r->next_elem = next_elem;
}
