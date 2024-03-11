#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "generator.h"

void switch_endianness(uint8_t* dest, const uint8_t* src, int length) {
    for (int i = 0; i < length; i++) {
        dest[length - 1 - i] = src[i];
    }
}

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

uint32_t random_uniform_n_lsb(rng* r, size_t n, uint8_t* batch) {
    uint32_t random_u32 = 0;
    for (int i = 0; i < 4; i++) {
        random_u32 |= (r->gen_rand_uniform(r, batch) << (i * 8));
    }

    return random_u32 >> (32 - n);
}

uint32_t gen_range(rng* r, size_t min, size_t max, uint8_t* batch) {
    if (min > max) {
        return -1;
    }

    size_t bit_len = (size_t) floor(log2(max - min));
    size_t a = min + random_uniform_n_lsb(r, bit_len, batch);

    while (a >= max) {
        a = min + random_uniform_n_lsb(r, bit_len, batch);
    }
    return a;
}

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

void copy_rng(rng* dest, const rng* src) {
    memcpy(dest->indices, src->indices, sizeof(src->indices));
    memcpy(dest->whitening, src->whitening, sizeof(src->whitening));
    dest->mode = src->mode;
    dest->gen_rand_uniform = src->gen_rand_uniform;
    dest->copy = src->copy;
    dest->next_elem = src->next_elem;
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

/*void rng_new_chacha(rng_chacha* r_cha, const uint8_t* seed_little_end, int mode) {
    int KEY_WIDTH = mode ? KEY_WIDTH_4 : KEY_WIDTH_B4;

    rng r;
    for (int i = 0; i < KEY_WIDTH; i++) {
        r.indices[i] = i;
    }
    r.mode = mode;
    r.gen_rand_uniform = random_uniform_chacha;

    r_cha->r = r;

    uint8_t batch[BATCH_SIZE];
    inc_and_new_batch_chacha();

    precompute_prng(&r_cha->r, batch);
}*/