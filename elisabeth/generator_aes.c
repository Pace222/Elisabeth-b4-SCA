#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "generator_aes.h"

#define BATCH_SIZE AES_BLOCKLEN * (1 + 5 * KEYROUND_WIDTH_4 / AES_BLOCKLEN)

void add_sub(uint8_t* array, size_t length, int carry) {
    for (int i = 0; i < length; i++) {
        int sum = array[i] + carry;
        array[i] = sum & 0xFF;
        carry = sum >> 8;
        if (carry == 0) break;
    }
}

void inc_and_new_batch_aes(uint8_t* batch, rng_aes* r_aes) {
    for (uint8_t* b = batch; b < batch + BATCH_SIZE; b += AES_BLOCKLEN) {
        memcpy(b, r_aes->ctr, AES_BLOCKLEN);
        AES_ECB_encrypt(&r_aes->ctx, b);
        add_sub(r_aes->ctr, AES_BLOCKLEN, 1);
    }
}

void recompute_last_batch_aes(uint8_t* batch, rng_aes* r_aes) {
    add_sub(r_aes->ctr, AES_BLOCKLEN, -BATCH_SIZE/AES_BLOCKLEN);
    inc_and_new_batch_aes(batch, r_aes);
}

uint8_t random_uniform_aes(rng* r, uint8_t* batch) {
    rng_aes* r_aes = (rng_aes*) r;

    if (r_aes->batch_idx >= BATCH_SIZE) {
        inc_and_new_batch_aes(batch, r_aes);
        r_aes->batch_idx = 0;
    }

    return batch[r_aes->batch_idx++];
}

void copy_aes(rng* dest, const rng* src) {
    rng_aes* dest_aes = (rng_aes*) dest;
    rng_aes* src_aes = (rng_aes*) src;

    copy_rng(&dest_aes->r, &src_aes->r);
    
    dest_aes->key = src_aes->key;
    dest_aes->batch_idx = src_aes->batch_idx;
    AES_init_ctx(&dest_aes->ctx, dest_aes->key);
    memcpy(dest_aes->ctr, src_aes->ctr, AES_BLOCKLEN);
}

void next_elem_aes(rng* r) {
    rng_aes* r_aes = (rng_aes*) r;

    uint8_t batch[BATCH_SIZE];
    recompute_last_batch_aes(batch, r_aes);

    precompute_prng(&r_aes->r, batch);
}

void rng_new_aes(rng_aes* r_aes, const uint8_t* seed_little_end, int mode) {
    int KEY_WIDTH = mode ? KEY_WIDTH_4 : KEY_WIDTH_B4;

    rng r;
    for (int i = 0; i < KEY_WIDTH; i++) {
        r.indices[i] = i;
    }
    r.mode = mode;
    r.gen_rand_uniform = random_uniform_aes;
    r.copy = copy_aes;
    r.next_elem = next_elem_aes;

    r_aes->r = r;
    r_aes->key = seed_little_end;
    AES_init_ctx(&r_aes->ctx, r_aes->key);
    memset(r_aes->ctr, 0, AES_BLOCKLEN);
    r_aes->batch_idx = 0;

    uint8_t batch[BATCH_SIZE];
    inc_and_new_batch_aes(batch, r_aes);

    precompute_prng(&r_aes->r, batch);
}
