#include <Profiler.h>


uint8_t seed_big_end[AES_BLOCKLEN], seed_little_end[AES_BLOCKLEN];
rng r;
uint32_t random_values_4[MAX_MESSAGE_SIZE][2 * KEYROUND_WIDTH_4];
uint32_t random_values_b4[MAX_MESSAGE_SIZE][2 * KEYROUND_WIDTH_B4];

uint4_t plaintext[MAX_MESSAGE_SIZE];

uint4_t keyround_block_4[BLOCK_WIDTH_4];
uint4_t keyround_4[KEYROUND_WIDTH_4];
uint4_t key_4[KEY_WIDTH_4];
uint4_t keyround_block_b4[BLOCK_WIDTH_B4];
uint4_t keyround_b4[KEYROUND_WIDTH_B4];
uint4_t key_b4[KEY_WIDTH_B4];

uint4_t ciphertext[MAX_MESSAGE_SIZE];

uint4_t decrypted[MAX_MESSAGE_SIZE];

uint4_t *keyround_block, *keyround, *key;
uint32_t *random_values;
int BLOCK_WIDTH, KEYROUND_WIDTH, KEY_WIDTH;
uint4_t filter_el, key_el;

String mode_str, choice;
int mode;

int actual_message_length;

int setup_mode() {
    mode_str = read_until(DELIMITER);
    if (mode_str == "4") {
      mode = 1;
      keyround_block = keyround_block_4;
      keyround = keyround_4;
      key = key_4;
      random_values = (uint32_t*) random_values_4;
      BLOCK_WIDTH = BLOCK_WIDTH_4;
      KEYROUND_WIDTH = KEYROUND_WIDTH_4;
      KEY_WIDTH = KEY_WIDTH_4;
      return 0;
    } else if (mode_str == "B4") {
      mode = 0;
      keyround_block = keyround_block_b4;
      keyround = keyround_b4;
      key = key_b4;
      random_values = (uint32_t*) random_values_b4;
      BLOCK_WIDTH = BLOCK_WIDTH_B4;
      KEYROUND_WIDTH = KEYROUND_WIDTH_B4;
      KEY_WIDTH = KEY_WIDTH_B4;
      return 0;
    } else {
      print_format(-1, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
      return 1;
    }
}

int setup_choice() {
    choice = read_until(DELIMITER);
    if (choice.length() <= 0) {
      print_format(mode, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
      return 1;
    }
    
    return 0;
}

void scenario_whitening_seed() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is the seed for the PRNG (IV), arg2 is the key. Output is the round key.
    if (fill_array_from_user_hex_bytes(seed_big_end, AES_BLOCKLEN, DELIMITER) != AES_BLOCKLEN || fill_array_from_user_hex(key, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(2 * AES_BLOCKLEN) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the round key (SIZE: " + String(KEYROUND_WIDTH) + " nibbles).");
        return;
    }

    switch_endianness(seed_little_end, seed_big_end, AES_BLOCKLEN);
    rng_new(&r, seed_little_end, mode);
    precompute_prng(random_values, &r);

    benchmark_whitening(keyround, key, &r, random_values);

    for (int i = 0; i < KEYROUND_WIDTH; i++) Serial.print(((char*) keyround)[i], HEX);
}

void scenario_filter_block() {
    // Format: [0-9A-Fa-f]. arg1 is a round key block. Output is the output of a single block of the filtering function.
    if (fill_array_from_user_hex(keyround_block, BLOCK_WIDTH, DELIMITER) != BLOCK_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is a round key block (SIZE: 1 + " + String(BLOCK_WIDTH) + " nibbles). Output is the output of a single block of the filtering function (SIZE: " + String(sizeof(filter_el)) + " nibbles).");
        return;
    }

    benchmark_filter_block(&filter_el, keyround_block, mode);

    Serial.print(filter_el, HEX);
}

void scenario_filter() {
    // Format: [0-9A-Fa-f]. arg1 is the round key. Output is the output of the filtering function.
    if (fill_array_from_user_hex(keyround, KEYROUND_WIDTH, DELIMITER) != KEYROUND_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the round key (SIZE: " + String(KEYROUND_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(key_el)) + " nibbles).");
        return;
    }

    benchmark_filter(&key_el, keyround, mode);

    Serial.print(key_el, HEX);
}

void scenario_whitening_and_filter() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is the seed for the PRNG (IV), arg2 is the key. Output is the output of the filtering function. 
    if (fill_array_from_user_hex_bytes(seed_big_end, AES_BLOCKLEN, DELIMITER) != AES_BLOCKLEN || fill_array_from_user_hex(key, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(2 * AES_BLOCKLEN) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(key_el)) + " nibbles).");
        return;
    }

    switch_endianness(seed_little_end, seed_big_end, AES_BLOCKLEN);
    rng_new(&r, seed_little_end, mode);
    precompute_prng(random_values, &r);

    benchmark_whitening_and_filter(&key_el, key, &r, random_values);

    Serial.print(key_el, HEX);
}

void scenario_addition() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is a single element of the plaintext, arg2 is the output of the filtering function. Output is a single element of the ciphertext.
    if (fill_array_from_user_hex(plaintext, 1, DELIMITER) != 1 || fill_array_from_user_hex(&key_el, 1, DELIMITER) != 1) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the plaintext (SIZE: " + String(1) + " nibbles), arg2 is the output of the filtering function (SIZE: " + String(1) + " nibbles). Output is a single element of the ciphertext (SIZE: " + String(1) + " nibbles).");
        return;
    }

    benchmark_addition(ciphertext, plaintext[0], key_el);

    Serial.print(ciphertext[0], HEX);
}

void scenario_subtraction() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is a single element of the ciphertext, arg2 is the output of the filtering function. Output is a single element of the plaintext.
    if (fill_array_from_user_hex(ciphertext, 1, DELIMITER) != 1 || fill_array_from_user_hex(&key_el, 1, DELIMITER) != 1) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the ciphertext (SIZE: " + String(1) + " nibbles), arg2 is the output of the filtering function (SIZE: " + String(1) + " nibbles). Output is a single element of the plaintext (SIZE: " + String(1) + " nibbles).");
        return;
    }

    benchmark_subtraction(decrypted, ciphertext[0], key_el);

    Serial.print(decrypted[0], HEX);
}

void scenario_encrypt_elem_seed() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f],[0-9A-Fa-f]. arg1 is the seed for the PRNG (IV), arg2 is a single element of the plaintext, arg3 is the key. Output is a single element of the ciphertext.
    if (fill_array_from_user_hex_bytes(seed_big_end, AES_BLOCKLEN, DELIMITER) != AES_BLOCKLEN || fill_array_from_user_hex(plaintext, 1, DELIMITER) != 1 || fill_array_from_user_hex(key, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(2 * AES_BLOCKLEN) + " nibbles), arg2 is a single element of the plaintext (SIZE: " + String(1) + " nibbles), arg3 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is a single element of the ciphertext (SIZE: " + String(1) + " nibbles).");
        return;
    }

    switch_endianness(seed_little_end, seed_big_end, AES_BLOCKLEN);
    rng_new(&r, seed_little_end, mode);
    precompute_prng(random_values, &r);

    benchmark_encrypt_element(ciphertext, plaintext[0], key, &r, random_values);

    Serial.print(ciphertext[0], HEX);
}

void scenario_decrypt_elem_seed() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f],[0-9A-Fa-f]. arg1 is the seed for the PRNG (IV), arg2 is a single element of the ciphertext, arg3 is the key. Output is a single element of the plaintext.
    if (fill_array_from_user_hex_bytes(seed_big_end, AES_BLOCKLEN, DELIMITER) != AES_BLOCKLEN || fill_array_from_user_hex(ciphertext, 1, DELIMITER) != 1 || fill_array_from_user_hex(key, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(2 * AES_BLOCKLEN) + " nibbles), arg2 is a single element of the ciphertext (SIZE: " + String(1) + " nibbles), arg3 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is a single element of the plaintext (SIZE: " + String(1) + " nibbles).");
        return;
    }

    switch_endianness(seed_little_end, seed_big_end, AES_BLOCKLEN);
    rng_new(&r, seed_little_end, mode);
    precompute_prng(random_values, &r);

    benchmark_decrypt_element(decrypted, ciphertext[0], key, &r, random_values);

    Serial.print(decrypted[0], HEX);
}

void scenario_encrypt_message_seed() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f],[0-9A-Fa-f]. arg1 is the seed for the PRNG (IV), arg2 is the complete plaintext (message) to encrypt, arg3 is the key. Output is the complete ciphertext (encrypted message).
    if (fill_array_from_user_hex_bytes(seed_big_end, AES_BLOCKLEN, DELIMITER) != AES_BLOCKLEN || (actual_message_length = fill_array_from_user_until(plaintext, DELIMITER)) <= 0 || fill_array_from_user_hex(key, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(2 * AES_BLOCKLEN) + " nibbles), arg2 is the complete plaintext (message) to encrypt (SIZE: max " + String(MAX_MESSAGE_SIZE) + " nibbles), arg3 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the complete ciphertext (encrypted message) (SIZE: same as plaintext).");
        return;
    }

    switch_endianness(seed_little_end, seed_big_end, AES_BLOCKLEN);
    rng_new(&r, seed_little_end, mode);
    precompute_prng(random_values, &r);

    {
        profiler_t p;
        encrypt(ciphertext, plaintext, key, &r, actual_message_length, random_values);
        for (int i = 0; i < AES_BLOCKLEN; i++) Serial.print(r.ctr[i], HEX);
    }

    for (int i = 0; i < actual_message_length; i++) Serial.print(((char*) ciphertext)[i], HEX);
}

void scenario_decrypt_message_seed() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f],[0-9A-Fa-f]. arg1 is the seed for the PRNG (IV), arg2 is the complete ciphertext (encrypted message) to decrypt, arg3 is the key. Output is the complete plaintext (decrypted message).
    if (fill_array_from_user_hex_bytes(seed_big_end, AES_BLOCKLEN, DELIMITER) != AES_BLOCKLEN || (actual_message_length = fill_array_from_user_until(ciphertext, DELIMITER)) <= 0 || fill_array_from_user_hex(key, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(2 * AES_BLOCKLEN) + " nibbles), arg2 is the complete ciphertext (encrypted message) to decrypt (SIZE: max " + String(MAX_MESSAGE_SIZE) + " nibbles), arg3 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the complete plaintext (decrypted message) (SIZE: same as ciphertext).");
        return;
    }

    switch_endianness(seed_little_end, seed_big_end, AES_BLOCKLEN);
    rng_new(&r, seed_little_end, mode);
    precompute_prng(random_values, &r);

    benchmark_decrypt_message(decrypted, ciphertext, key, &r, actual_message_length, random_values);

    for (int i = 0; i < actual_message_length; i++) Serial.print(((char*) decrypted)[i], HEX);
}
