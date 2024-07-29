#include "generator_aes.h"
#include "generator_cha.h"
#include "delays.h"
#include "encryption.h"

#define interruptPIN A1                        /* Interrupt PIN */
#define TriggerPQ A0                           /* Trigger PIN */
#define TRIGGER_DELAY 20                       /* Time in microseconds for the trigger to stay LOW before setting it to HIGH */

#define DEVICE_SECRET_SIZE AES_KEYLEN          /* Size of the target secret, if using our novel countermeasure */

#define MAX_MESSAGE_SIZE 16                    /* Maximum number of elements per message */
#define MAX_INPUT_SIZE (32 + AES_KEYLEN + MAX_MESSAGE_SIZE + KEY_WIDTH_B4) /* Maximum size of input command */
#define START '<'                              /* Character signifiying the start of a command */
#define DELIMITER ','                          /* Character signifiying the delimitation of arguments */
#define STOP '>'                               /* Character signifiying the end of a command */

int new_data;
char input_buffer[MAX_INPUT_SIZE + 1];
size_t process_head;

struct aes_or_cha_list {
  int type;                                    /* 1 for ChaCha20, 0 for AES */
  union {
    rng_aes aes[MAX_MESSAGE_SIZE];
    rng_cha cha[MAX_MESSAGE_SIZE];
  } l;
};

uint8_t buf_seed_1[AES_KEYLEN], buf_seed_2[AES_KEYLEN];
uint4_t buf_message[MAX_MESSAGE_SIZE], buf_out[MAX_MESSAGE_SIZE];
packed buf_out_shares[MAX_MESSAGE_SIZE];
uint4_t buf_arg[KEY_WIDTH_B4];
packed buf_shares[KEY_WIDTH_B4];
aes_or_cha_list rng_list;
rng* chosen_rng;
const rng* rng_refs[MAX_MESSAGE_SIZE];

int BLOCK_WIDTH, KEYROUND_WIDTH, KEY_WIDTH;

String mode_str, repeat_str, choice;
int mode, repeat;
size_t actual_message_length;

/**
 * \brief          Set the trigger for the oscilloscope before running RWS
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     keyround: Output variable used to store the keyround
 * \param[in]      key: The key
 * \param[in]      r: The instance of a PRNG used in the RWS
 */
void benchmark_whitening(uint4_t* keyround, uint4_t* key, rng* r) {
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);

  random_whitened_subset(keyround, key, r);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  interrupts();
}

/**
 * \brief          Scenario for RWS, expects one argument: the key. Result is the keyround
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_whitening() {
    if (fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the keyround (SIZE: " + String(KEYROUND_WIDTH) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_whitening(buf_out, buf_arg, chosen_rng);

      /* Send back the operation result */
      for (int i = 0; i < KEYROUND_WIDTH; i++) Serial.print(((char*) buf_out)[i], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before running one round of the filtering function
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     filter_el: Output variable used to store the output of the round
 * \param[in]      block: The input block to the round 
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 */
void benchmark_filter_block(uint4_t* filter_el, uint4_t* block, int mode) {
  uint4_t (*filter_block) (const uint4_t*) = mode ? filter_block_4 : filter_block_b4;

  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  uint4_t res = filter_block(block);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  delayMicroseconds(25);
  interrupts();

  *filter_el = res;
}

/**
 * \brief          Scenario for one round, expects one argument: the keyround block. Result is the round's output
 */
void scenario_filter_block() {
    if (fill_array_from_user_hex(buf_arg, BLOCK_WIDTH, DELIMITER) != BLOCK_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is a keyround block (SIZE: 1 + " + String(BLOCK_WIDTH) + " nibbles). Output is the output of a single block of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles).");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_filter_block(buf_out, buf_arg, mode);

      /* Send back the operation result */
      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before running the 14 rounds of the filtering function with the entire keyround
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     key_el: Output variable used to store the keystream
 * \param[in]      keyround: The input keyround
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4
 */
void benchmark_filter(uint4_t* key_el, uint4_t* keyround, int mode) {
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  uint4_t res = filter(keyround, mode);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  interrupts();

  *key_el = res;
}

/**
 * \brief          Scenario for 14 rounds, expects one argument: the keyround. Result is the keystream
 */
void scenario_filter() {
    if (fill_array_from_user_hex(buf_arg, KEYROUND_WIDTH, DELIMITER) != KEYROUND_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the keyround (SIZE: " + String(KEYROUND_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles).");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_filter(buf_out, buf_arg, mode);

      /* Send back the operation result */
      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before running the complete algorithm that generates the keystream: RWS and the 14 rounds
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     key_el: Output variable used to store the output the keystream
 * \param[in]      key: The key
 * \param[in]      r: The instance of a PRNG used in the RWS
 */
void benchmark_whitening_and_filter(uint4_t* key_el, uint4_t* key, rng* r) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  random_whitened_subset(keyround, key, r);
  uint4_t res = filter(keyround, r->mode);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  delayMicroseconds(50);
  interrupts();

  *key_el = res;
}

/**
 * \brief          Scenario for RWS + 14 rounds, expects one argument: the key. Result is the keystream
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_whitening_and_filter() {
    if (fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_whitening_and_filter(buf_out, buf_arg, chosen_rng);

      /* Send back the operation result */
      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before running the complete algorithm that generates the keystream: RWS and the 14 rounds,
 *                 protected with 2-share arithmetic masking
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     key_el: Output variable used to store the output the keystream
 * \param[in]      key: The key
 * \param[in]      r: The instance of a PRNG used in the RWS
 */
void benchmark_masked_whitening_and_filter(packed* key_el, packed* key, rng* r) {
  packed keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  masked_random_whitened_subset(keyround, key, r);
  packed res = masked_filter(keyround, r->mode);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  delayMicroseconds(50);
  interrupts();

  *key_el = res;
}

/**
 * \brief          Scenario for 2-share arithmetic masked RWS + 14 rounds, expects one argument: the key. Result is the keystream's mask shares
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_masked_whitening_and_filter() {
    if (fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    /* Generate random mask shares for the key */
    init_shares(buf_shares, buf_arg, KEY_WIDTH);
    
    /* Prefill internal mask shares */
    generate_masking_random_table();
    for (int i = 0; i < repeat; i++) {
      reset_masking_counter();
      benchmark_masked_whitening_and_filter(buf_out_shares, buf_shares, chosen_rng);

      /* Send back the operation result */
      Serial.print(SHARE_0(buf_out_shares[0]), HEX);
      Serial.print(SHARE_1(buf_out_shares[0]), HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Scenario for 2-share arithmetic masked RWS + 14 rounds with one of the shares being 0, expects one argument: the key. Result is the keystream's mask shares, one of them being 0
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_masked_null_whitening_and_filter() {
    if (fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    /* Generate null mask shares for the key */
    init_null_shares(buf_shares, buf_arg, KEY_WIDTH);

    /* Prefill null internal mask shares */
    generate_null_masking_random_table();
    for (int i = 0; i < repeat; i++) {
      reset_masking_counter();
      benchmark_masked_whitening_and_filter(buf_out_shares, buf_shares, chosen_rng);

      /* Send back the operation result */
      Serial.print(SHARE_0(buf_out_shares[0]), HEX);
      Serial.print(SHARE_1(buf_out_shares[0]), HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before running the complete algorithm that generates the keystream: RWS and the 14 rounds,
 *                 protected with shuffling
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     key_el: Output variable used to store the output the keystream
 * \param[in]      key: The key
 * \param[in]      r: The instance of a PRNG used in the RWS
 */
void benchmark_shuffled_whitening_and_filter(uint4_t* key_el, uint4_t* key, rng* r) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  shuffled_random_whitened_subset(keyround, key, r);
  uint4_t res = shuffled_filter(keyround, r->mode);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  delayMicroseconds(50);
  interrupts();

  *key_el = res;
}

/**
 * \brief          Scenario for shuffled RWS + 14 rounds, expects one argument: the key. Result is the keystream
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_shuffled_whitening_and_filter() {
    if (fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    /* Prefill RSIs */
    generate_shuffling_random_table();
    for (int i = 0; i < repeat; i++) {
      reset_masking_counter();
      benchmark_shuffled_whitening_and_filter(buf_out, buf_arg, chosen_rng);

      /* Send back the operation result */
      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before running the complete algorithm that generates the keystream: RWS and the 14 rounds,
 *                 protected with 2-share arithmetic masking and shuffling
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     key_el: Output variable used to store the output the keystream
 * \param[in]      key: The key
 * \param[in]      r: The instance of a PRNG used in the RWS
 */
void benchmark_masked_shuffled_whitening_and_filter(packed* key_el, packed* key, rng* r) {
  packed keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  masked_shuffled_random_whitened_subset(keyround, key, r);
  packed res = masked_shuffled_filter(keyround, r->mode);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  delayMicroseconds(50);
  interrupts();

  *key_el = res;
}

/**
 * \brief          Scenario for 2-share arithmetic masked and shuffled RWS + 14 rounds, expects one argument: the key. Result is the keystream's mask shares
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_masked_shuffled_whitening_and_filter() {
    if (fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    /* Generate random mask shares for the key */
    init_shares(buf_shares, buf_arg, KEY_WIDTH);

    /* Prefill internal mask shares */
    generate_masking_shuffling_random_table();
    for (int i = 0; i < repeat; i++) {
      reset_masking_counter();
      benchmark_masked_shuffled_whitening_and_filter(buf_out_shares, buf_shares, chosen_rng);

      /* Send back the operation result */
      Serial.print(SHARE_0(buf_out_shares[0]), HEX);
      Serial.print(SHARE_1(buf_out_shares[0]), HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before running the complete algorithm that generates the keystream: RWS and the 14 rounds,
 *                 protected with our novel countermeasure, using delays
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     key_el: Output variable used to store the output the keystream
 * \param[in]      key: The key
 * \param[in]      r: The instance of a PRNG used in the RWS
 */
void benchmark_whitening_and_filter_delayed(uint4_t* key_el, uint4_t* key, rng* r) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  random_whitened_subset_delayed(keyround, key, r);
  uint4_t res = filter_delayed(keyround, r->mode);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  delayMicroseconds(50);
  interrupts();

  *key_el = res;
}

/**
 * \brief          Scenario for RWS + 14 rounds protected with our novel countermeasure, expects one argument: the key. Result is the keystream
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_whitening_and_filter_delayed() {
    if (fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    /* Precompute and print delays */
    new_encryption(buf_arg);
    print_delays();

    for (int i = 0; i < repeat; i++) {
      new_encryption(buf_arg);
      benchmark_whitening_and_filter_delayed(buf_out, buf_arg, chosen_rng);

      /* Send back the operation result */
      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before running the addition of the keystream with the plaintext
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     cipher_el: Output variable used to store the ciphertext
 * \param[in]      plain_el: The plaintext
 * \param[in]      key_el: The keystream
 */
void benchmark_addition(uint4_t* cipher_el, uint4_t plain_el, uint4_t key_el) {
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  uint4_t r = uint4_add(plain_el, key_el);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  interrupts();

  *cipher_el = r;
}

/**
 * \brief          Scenario for the addition of the keystream with the plaintext, expects two arguments: the plaintext and the keystream. Result is the ciphertext
 */
void scenario_addition() {
    if (fill_array_from_user_hex(buf_message, 1, DELIMITER) != 1 || fill_array_from_user_hex(buf_arg, 1, DELIMITER) != 1) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the plaintext (SIZE: 1 + " + String(1) + " nibbles), arg2 is the output of the filtering function (SIZE: 1 + " + String(1) + " nibbles). Output is a single element of the ciphertext (SIZE: " + String(1) + " nibbles).");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_addition(buf_out, buf_message[0], buf_arg[0]);

      /* Send back the operation result */
      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before running the subtraction of the keystream from the ciphertext
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     decrypted_el: Output variable used to store the decrypted ciphertext
 * \param[in]      cipher_el: The ciphertext
 * \param[in]      key_el: The keystream
 */
void benchmark_subtraction(uint4_t* decrypted_el, uint4_t cipher_el, uint4_t key_el) {
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  uint4_t r = uint4_add(cipher_el, uint4_neg(key_el));

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  interrupts();

  *decrypted_el = r;
}

/**
 * \brief          Scenario for the subtraction of the keystream from the ciphertext, expects two arguments: the ciphertext and the keystream. Result is the decrypted ciphertext
 */
void scenario_subtraction() {
    if (fill_array_from_user_hex(buf_message, 1, DELIMITER) != 1 || fill_array_from_user_hex(buf_arg, 1, DELIMITER) != 1) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the ciphertext (SIZE: 1 + " + String(1) + " nibbles), arg2 is the output of the filtering function (SIZE: 1 + " + String(1) + " nibbles). Output is a single element of the plaintext (SIZE: " + String(1) + " nibbles).");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_subtraction(buf_out, buf_message[0], buf_arg[0]);

      /* Send back the operation result */
      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before encrypting a plaintext: RWS, 14 rounds and final addition
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     cipher_el: Output variable used to store the ciphertext
 * \param[in]      plain_el: The plaintext
 * \param[in]      key: The key
 * \param[in]      r: The instance of a PRNG used in the RWS
 */
void benchmark_encrypt_element(uint4_t* cipher_el, uint4_t plain_el, uint4_t* key, rng* r) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];

  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);

  random_whitened_subset(keyround, key, r);
  uint4_t res = uint4_add(plain_el, filter(keyround, r->mode));

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  interrupts();

  *cipher_el = res;
}

/**
 * \brief          Scenario for the plaintext encryption (RWS + 14 rounds + addition), expects two arguments: the plaintext and the key. Result is the ciphertext
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_encrypt_elem() {
    if (fill_array_from_user_hex(buf_message, 1, DELIMITER) != 1 || fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the plaintext (SIZE: 1 + " + String(1) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is a single element of the ciphertext (SIZE: " + String(1) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_encrypt_element(buf_out, buf_message[0], buf_arg, chosen_rng);

      /* Send back the operation result */
      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before decrypting a ciphertext: RWS, 14 rounds and final subtraction
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     decrypted_el: Output variable used to store the decrypted ciphertext
 * \param[in]      cipher_el: The ciphertext
 * \param[in]      key: The key
 * \param[in]      r: The instance of a PRNG used in the RWS
 */
void benchmark_decrypt_element(uint4_t* decrypted_el, uint4_t cipher_el, uint4_t* key, rng* r) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];

  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);

  random_whitened_subset(keyround, key, r);
  uint4_t res = uint4_add(cipher_el, uint4_neg(filter(keyround, r->mode)));

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  interrupts();

  *decrypted_el = res;
}

/**
 * \brief          Scenario for the plaintext decryption (RWS + 14 rounds + subtraction), expects two arguments: the ciphertext and the key. Result is the decrypted ciphertext
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_decrypt_elem() {
    if (fill_array_from_user_hex(buf_message, 1, DELIMITER) != 1 || fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the ciphertext (SIZE: 1 + " + String(1) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is a single element of the plaintext (SIZE: " + String(1) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_decrypt_element(buf_out, buf_message[0], buf_arg, chosen_rng);

      /* Send back the operation result */
      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before encrypting a message: RWS, 14 rounds and final addition for each plaintext element
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     ciphertext: Output variable used to store the ciphertext message
 * \param[in]      plaintext: The plaintext message
 * \param[in]      key: The key
 * \param[in]      r: The instances of the PRNGs used in the RWS, one for each element to encrypt
 * \param[in]      length: The number of elements in the message
 */
void benchmark_encrypt_message(uint4_t* ciphertext, uint4_t* plaintext, uint4_t* key, const rng** r, size_t length) {
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  encrypt(ciphertext, plaintext, key, r, length);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  interrupts();
}

/**
 * \brief          Scenario for the message encryption (RWS + 14 rounds + addition for each element), expects two arguments: the plaintext message and the key. Result is the ciphertext message
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_encrypt_message() {
    if ((actual_message_length = fill_array_from_user_until(buf_message, DELIMITER)) <= 0 || fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the complete plaintext (message) to encrypt (SIZE: max " + String(MAX_MESSAGE_SIZE) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the complete ciphertext (encrypted message) (SIZE: same as plaintext). Expects to have filled the random table in a previous command.");
        return;
    }

    /* Precompute the PRNGs for every element of the message */
    switch (rng_list.type) {
      case 0:                                  /* AES */
        rng_refs[0] = &rng_list.l.aes[0].r;
        for (int i = 1; i < actual_message_length; i++) {
          rng_list.l.aes[i - 1].r.copy(&rng_list.l.aes[i].r, &rng_list.l.aes[i - 1].r);
          rng_list.l.aes[i].r.next_elem(&rng_list.l.aes[i].r);
          rng_refs[i] = &rng_list.l.aes[i].r;
        }
        break;
      case 1:                                  /* ChaCha20 */
        rng_refs[0] = &rng_list.l.cha[0].r;
        for (int i = 1; i < actual_message_length; i++) {
          rng_list.l.cha[i - 1].r.copy(&rng_list.l.cha[i].r, &rng_list.l.cha[i - 1].r);
          rng_list.l.cha[i].r.next_elem(&rng_list.l.cha[i].r);
          rng_refs[i] = &rng_list.l.cha[i].r;
        }
        break;
      default:
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the complete plaintext (message) to encrypt (SIZE: max " + String(MAX_MESSAGE_SIZE) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the complete ciphertext (encrypted message) (SIZE: same as plaintext). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_encrypt_message(buf_out, buf_message, buf_arg, rng_refs, actual_message_length);

      /* Send back the operation result */
      for (int i = 0; i < actual_message_length; i++) Serial.print(((char*) buf_out)[i], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before decrypting a message: RWS, 14 rounds and final subtraction for each ciphertext element
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     decrypted: Output variable used to store the decrypted ciphertext message
 * \param[in]      ciphertext: The ciphertext message
 * \param[in]      key: The key
 * \param[in]      r: The instances of the PRNGs used in the RWS, one for each element to decrypt
 * \param[in]      length: The number of elements in the message
 */
void benchmark_decrypt_message(uint4_t* decrypted, uint4_t* ciphertext, uint4_t* key, const rng** r, size_t length) {
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  decrypt(decrypted, ciphertext, key, r, length);

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  interrupts();
}

/**
 * \brief          Scenario for the message decryption (RWS + 14 rounds + subtraction for each element), expects two arguments: the ciphertext message and the key. Result is the decrypted ciphertext message
 * \note           It is expected that the random table has been previously filled with the ``genRndChacha'' or ``genRndAES'' command
 */
void scenario_decrypt_message() {
    if ((actual_message_length = fill_array_from_user_until(buf_message, DELIMITER)) <= 0 || fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the complete ciphertext (encrypted message) to decrypt (SIZE: max " + String(MAX_MESSAGE_SIZE) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the complete plaintext (decrypted message) (SIZE: same as ciphertext). Expects to have filled the random table in a previous command.");
        return;
    }

    /* Precompute the PRNGs for every element of the message */
    switch (rng_list.type) {
      case 0:                                  /* AES */
        rng_refs[0] = &rng_list.l.aes[0].r;
        for (int i = 1; i < actual_message_length; i++) {
          rng_list.l.aes[i - 1].r.copy(&rng_list.l.aes[i].r, &rng_list.l.aes[i - 1].r);
          rng_list.l.aes[i].r.next_elem(&rng_list.l.aes[i].r);
          rng_refs[i] = &rng_list.l.aes[i].r;
        }
        break;
      case 1:                                  /* ChaCha20 */
        rng_refs[0] = &rng_list.l.cha[0].r;
        for (int i = 1; i < actual_message_length; i++) {
          rng_list.l.cha[i - 1].r.copy(&rng_list.l.cha[i].r, &rng_list.l.cha[i - 1].r);
          rng_list.l.cha[i].r.next_elem(&rng_list.l.cha[i].r);
          rng_refs[i] = &rng_list.l.cha[i].r;
        }
        break;
      default:
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the complete plaintext (message) to encrypt (SIZE: max " + String(MAX_MESSAGE_SIZE) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the complete ciphertext (encrypted message) (SIZE: same as plaintext). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_decrypt_message(buf_out, buf_message, buf_arg, rng_refs, actual_message_length);

      /* Send back the operation result */
      for (int i = 0; i < actual_message_length; i++) Serial.print(((char*) buf_out)[i], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
      Serial.flush();
    }
}

/**
 * \brief          Set the trigger for the oscilloscope before applying a S-Box
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     output: Output variable used to store the result of the S-Box
 * \param[in]      index: The S-Box we want to evaluate
 * \param[in]      input: The input block of the corresponding block
 */
void benchmark_test_sbox(uint4_t* output, uint4_t index, uint4_t* input) {
  noInterrupts();
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);  

  uint4_t res = S_BOXES_B4[index][input[index]];

  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(TRIGGER_DELAY);
  digitalWrite(TriggerPQ, HIGH);
  interrupts();

  *output = res;
}

/**
 * \brief          Scenario for the S-Box evaluation, expects two arguments: the S-Box index and the input to the S-Box. Result is the S-Box output
 */
void scenario_test_sbox() {
  if (fill_array_from_user_hex(buf_seed_1, 1, DELIMITER) != 1 || fill_array_from_user_hex(buf_arg, 1, DELIMITER) != 1 || buf_seed_1[0] < 0 || buf_seed_1[0] > 5) {
    print_format(mode, choice, "[0-5],[0-9A-Fa-f]", "arg1 is the S-Box index (SIZE: 1 + " + String(1) + " nibbles), arg2 is the input to the S-Box (SIZE: 1 + " + String(1) + " nibbles). Output is the output of the S-Box.");
    return;
  }

  buf_arg[buf_seed_1[0]] = buf_arg[0];

  for (int i = 0; i < repeat; i++) {
    benchmark_test_sbox(buf_out, buf_seed_1[0], buf_arg);

    /* Send back the operation result */
    Serial.print(buf_out[0], HEX);
    if (i < repeat - 1) {
      Serial.print(DELIMITER);
    }
    Serial.flush();
  }
}

/**
 * \brief          Scenario for precomputing the PRNG using AES, expects one argument: the seed (IV). No result
 */
void scenario_fill_rnd_table_aes() {
  if (fill_array_from_user_hex_bytes(buf_seed_1, AES_KEYLEN, DELIMITER) != AES_KEYLEN) {
      print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(AES_KEYLEN) + " bytes). No output.");
      return;
  }

  rng_list.type = 0;

  switch_endianness(buf_seed_2, buf_seed_1, AES_KEYLEN);
  rng_new_aes(&rng_list.l.aes[0], buf_seed_2, mode);

  chosen_rng = &rng_list.l.aes[0].r;
}

/**
 * \brief          Scenario for precomputing the PRNG using ChaCha20, expects one argument: the seed (IV). No result
 */
void scenario_fill_rnd_table_chacha() {
  if (fill_array_from_user_hex_bytes(buf_seed_1, CHACHA_KEYLEN, DELIMITER) != CHACHA_KEYLEN) {
      print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(CHACHA_KEYLEN) + " bytes). No output.");
      return;
  }

  rng_list.type = 1;

  switch_endianness(buf_seed_2, buf_seed_1, CHACHA_KEYLEN);
  rng_new_cha(&rng_list.l.cha[0], buf_seed_2, mode);

  chosen_rng = &rng_list.l.cha[0].r;
}

/**
 * \brief          Scenario for setting the device secret as part of our novel countermeasure, expects one argument: the secret. No result
 */
void scenario_set_device_secret() {
  if (fill_array_from_user_hex_bytes(buf_seed_1, DEVICE_SECRET_SIZE, DELIMITER) != DEVICE_SECRET_SIZE) {
      print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the device secret as part of our novel countermeasure (SIZE: " + String(DEVICE_SECRET_SIZE) + " bytes). No output.");
      return;
  }

  init_chain(buf_seed_1, DEVICE_SECRET_SIZE);
}


/**
 * \brief          Turns character into its decimal interpretation
 * \param[in]      ch: The input character
 * \return         The decimal representation, -1 if not possible
 */
int hex2int(char ch) {
  if (ch >= '0' && ch <= '9')
      return ch - '0';
  if (ch >= 'A' && ch <= 'F')
      return ch - 'A' + 10;
  if (ch >= 'a' && ch <= 'f')
      return ch - 'a' + 10;
  return -1;
}

/**
 * \brief          Fills the target array with random values
 * \note           This function does not return a value, it stores it to a pointer instead
 * \param[out]     array: The target array
 * \param[in]      length: The length of the target array
 */
void fill_random_uint4_t(uint4_t* array, size_t length) {
  for (size_t i = 0; i < length; i++) {
    array[i] = uint4_new((uint8_t) rand());
  }
}

/**
 * \brief          Reads the input buffer until the next delimiter
 * \param[in]      delimiter: The character until which we stop reading
 * \return         A String of data until the next delimiter (excluded), an empty string otherwise
 */
String read_until(char delimiter) {
  char* start = input_buffer + process_head;
  char c = input_buffer[process_head++];
  while (c != '\0' && c != delimiter) {        /* Read until the buffer is empty or the delimiter is reached */
    c = input_buffer[process_head++]; 
  }

  if (c == delimiter) {
    input_buffer[process_head - 1] = '\0';     /* End the char array for the subsequent call to String() */
    return String(start);
  } else {
    /* Delimiter was not found, fix the head back to the NULL character and return an empty string */
    process_head -= 1;
    return String();
  }
}

/**
 * \brief          Fills the target array with a given number of input bytes. There should be the delimiter directly afterwards
 * \param[out]     array: The target array containing bytes expressed as two hexadecimal numbers
 * \param[in]      length: The number of inputs to read
 * \param[in]      delimiter: The delimiter present after the input values
 * \return         The number of characters read, -1 if wrong input or if `length` values have been read but `delimiter` is not following them
 */
int fill_array_from_user_hex_bytes(uint8_t* array, size_t length, char delimiter) {
  char c1, c2;
  int n1, n2;

  /* Skip first nibble if we need an odd number of nibbles. */
  if (length % 2 == 1) {
    c1 = input_buffer[process_head];
    if (c1 == '\0' || hex2int(c1) < 0) {
      return 0;
    } else {
      process_head += 1;
    }
  }

  for (int i = 0; i < length; i++) {
    /* One byte = Two nibbles */
    c1 = input_buffer[process_head];
    c2 = input_buffer[process_head + 1];

    if (c1 == '\0' || c2 == '\0') {
      /* Buffer is empty */
      return i;
    } else {
      n1 = hex2int(c1);
      n2 = hex2int(c2);
      if (n1 < 0 || n2 < 0) {
        /* Wrong input: character is not an hexadecimal number */
        return i;
      }

      array[i] = (n1 << 4) + n2;               /* Store the byte */

      process_head += 2;
    }
  }

  c1 = input_buffer[process_head];
  /* Following character should be the delimiter or the end of the buffer */
  if (c1 == delimiter) {
    process_head += 1;                         /* Skip the delimiter for further parsing */
    return length;
  } else if (c1 == '\0') {
    return length;
  } else {
    return -1;
  }
}

/**
 * \brief          Fills the target array with a given number of input nibbles. There should be the delimiter directly afterwards
 * \param[out]     array: The target array containing nibbles expressed as one hexadecimal number
 * \param[in]      length: The number of inputs to read
 * \param[in]      delimiter: The delimiter present after the input values
 * \return         The number of characters read, -1 if wrong input or if `length` values have been read but `delimiter` is not following them
 */
int fill_array_from_user_hex(uint4_t* array, size_t length, char delimiter) {
  char c;
  int n;

  /* Skip first nibble if we need an odd number of nibbles. */
  if (length % 2 == 1) {
    c = input_buffer[process_head];
    if (c == '\0' || hex2int(c) < 0) {
      return 0;
    } else {
      process_head += 1;
    }
  }

  for (int i = 0; i < length; i++) {
    c = input_buffer[process_head];

    if (c == '\0') {
      /* Buffer is empty */
      return i;
    } else {
      n = hex2int(c);
      if (n < 0) {
        /* Wrong input: character is not an hexadecimal number */
        return i;
      }

      array[i] = uint4_new(n);                 /* Store the byte */

      process_head += 1;
    }
  }

  c = input_buffer[process_head];
  /* Following character should be the delimiter or the end of the buffer */
  if (c == delimiter) {
    process_head += 1;                         /* Skip the delimiter for further parsing */
    return length;
  } else if (c == '\0') {
    return length;
  } else {
    return -1;
  }
}

/**
 * \brief          Fills the target array with input nibbles until the delimiter is found
 * \param[out]     array: The target array containing nibbles expressed as one hexadecimal number
 * \param[in]      delimiter: The delimiter after which we stop reading
 * \return         The number of characters read, -1 if wrong input or if the maximum message size is reached
 */
int fill_array_from_user_until(uint4_t* array, char delimiter) {
  char c;
  int n;
  int length;

  for (length = 0; length < MAX_MESSAGE_SIZE; length++) {
    c = input_buffer[process_head];
    if (c == '\0') {
      /* Buffer is empty */
      return length;
    } else if (c == delimiter) {
      /* Delimiter found, stop reading */
      process_head += 1; /* Skip the delimiter for further parsing */
      return length;
    } else {
      n = hex2int(c);
      if (n < 0) {
        /* Wrong input: character is not an hexadecimal number */
        return -1;
      }

      array[length] = uint4_new(n);            /* Store the byte */

      process_head += 1;
    }
  }

  c = input_buffer[process_head];
  /* Following character should be the delimiter or the end of the buffer */
  if (c == delimiter) {
    process_head += 1;                         /* Skip the delimiter for further parsing */
    return length;
  } else if (c == '\0') {
    return length;
  } else {
    return -1;
  }
}

/**
 * \brief          Send back ("print") error messages with different formats
 * \param[in]      mode: 1 for Elisabeth-4, 0 for Elisabeth-b4, else for generic
 * \param[in]      choice: The chosen scenario
 * \param[in]      arguments: The format of the corresponding scenario's arguments
 * \param[in]      description: The description of the scenario
 */
void print_format(int mode, String choice, String arguments, String description) {
  String str_to_print = "Format: " + String(START);

  if (mode == 1) {
    str_to_print += "[4]" + String(DELIMITER);
  } else if (mode == 0) {
    str_to_print += "[B4]" + String(DELIMITER);
  } else {
    str_to_print += "[4 | B4]" + String(DELIMITER);
  }

  str_to_print += "[0-9]" + String(DELIMITER);

  if (choice != "\0") {
    str_to_print += "[" + choice + "]" + String(DELIMITER);
  } else {
    str_to_print += "[CHOICE]" + String(DELIMITER);
  }

  str_to_print += arguments + String(STOP) + " | Elisabeth-4/b4 | Number of repetitions | Scenario | " + description;

  str_to_print += " All arguments must be in hexadecimal format, except for the number of repetitions which is in decimal.";

  Serial.println(str_to_print);

  if (choice == "\0") {
    Serial.println("  0 : Benchmark whitening");
    Serial.println("  1 : Benchmark single block of filter function");
    Serial.println("  2 : Benchmark full filter function");
    Serial.println("  3 : Benchmark whitening + full filter function");
    Serial.println("  3p: Benchmark whitening + full filter function with side-channel defenses enabled");
    Serial.println("  4 : Benchmark final addition with plaintext (encryption)");
    Serial.println("  5 : Benchmark final subtraction with ciphertext (decryption)");
    Serial.println("  6 : Benchmark complete encryption, single element");
    Serial.println("  7 : Benchmark complete decryption, single element");
    Serial.println("  8 : Benchmark complete encryption, full message");
    Serial.println("  9 : Benchmark complete decryption, full message");
    Serial.println("  genRndAES   : Fill a table with random values for faster subsequent lookups with AES");
    Serial.println("  genRndChacha: Fill a table with random values for faster subsequent lookups with Chacha");
    Serial.println("  testSBox    : Run a given S-Box to test it");
  }
  Serial.flush();
}

/**
 * \brief          Read from the user and store inside the buffer for further parsing
 * \note           This function is executed continuously along `process_input` inside the `loop` function
 */
void recv_input() {
  static int recv_in_progress = 0;
  static int truncated = 0;
  static size_t head = 0;
  char c;

  while (Serial.available() > 0 && !new_data) {
    /* Character by character */
    c = Serial.read();

    /* Ignore any character not between `START` and `STOP` */
    if (recv_in_progress) {
      if (c != STOP) {
        input_buffer[head] = c;                /* Add character to buffer */
        head++;
        if (head > MAX_INPUT_SIZE) {
          /* If too long, overwrite last character with new one */
          head = MAX_INPUT_SIZE;
          truncated = 1;
        }
      } else {
        input_buffer[head] = '\0';             /* Terminate the string */
        recv_in_progress = 0;  
        head = 0;
        if (truncated) {
          /* If input was too long, discard it */
          Serial.println("Input too long.");
          print_format(-1, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
          truncated = 0;
        } else {
          /* Else, indicate it with `new_data` */
          new_data = 1;
        }
      }
    } else if (c == START) {
      recv_in_progress = 1;
    }
  }
}

/**
 * \brief          Read the mode and set the appropriate variables accordingly
 * \return         0 for OK, 1 for error
 */
int setup_mode() {
    mode_str = read_until(DELIMITER);
    if (mode_str == "4") {
      mode = 1;
      BLOCK_WIDTH = BLOCK_WIDTH_4;
      KEYROUND_WIDTH = KEYROUND_WIDTH_4;
      KEY_WIDTH = KEY_WIDTH_4;
      return 0;
    } else if (mode_str == "B4") {
      mode = 0;
      BLOCK_WIDTH = BLOCK_WIDTH_B4;
      KEYROUND_WIDTH = KEYROUND_WIDTH_B4;
      KEY_WIDTH = KEY_WIDTH_B4;
      return 0;
    } else {
      print_format(-1, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
      return 1;
    }
}

/**
 * \brief          Read the number of times to repeat the scenario and set the appropriate variables accordingly
 * \return         0 for OK, 1 for error
 */
int setup_repeat() {
    repeat_str = read_until(DELIMITER);
    repeat = repeat_str.toInt();
    if (repeat <= 0) {
      print_format(mode, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
      return 1;
    }
    
    return 0;
}

/**
 * \brief          Read the scenario choice and set the appropriate variables accordingly
 * \return         0 for OK, 1 for error
 */
int setup_choice() {
    choice = read_until(DELIMITER);
    if (choice.length() == 0) {
      print_format(mode, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
      return 1;
    }
    
    return 0;
}

/**
 * \brief          Executes the scenario with given arguments, according to the input.
 */
void process_input() {
  if (new_data) {
    /* Process data when it arrives and reset the corresponding variables */
    process_head = 0;
    new_data = 0;

    if (setup_mode()) {
      return;
    }
    
    if (setup_repeat()) {
      return;
    }

    if (setup_choice()) {
      return;
    }

    if (choice == "0") {
      /* Benchmark whitening */
      scenario_whitening();
    } else if (choice == "1") {
      /* Benchmark single block of filter function */
      scenario_filter_block();
    } else if (choice == "2") {
      /* Benchmark full filter function */
      scenario_filter();
    } else if (choice == "3") {
      /* Benchmark whitening + full filter function */
      scenario_whitening_and_filter();
    } else if (choice == "3m") {
      /* Benchmark masked whitening + full filter function */
      scenario_masked_whitening_and_filter();
    } else if (choice == "3m0") {
      /* Benchmark masked whitening + full filter function with empty masks */
      scenario_masked_null_whitening_and_filter();
    } else if (choice == "3s") {
      /* Benchmark shuffled whitening + full filter function */
      scenario_shuffled_whitening_and_filter();
    } else if (choice == "3ms") {
      /* Benchmark masked and shuffled whitening + full filter function */
      scenario_masked_shuffled_whitening_and_filter();
    } else if (choice == "3d") {
      /* Benchmark delayed whitening + full filter function (novel countermeasure) */
      scenario_whitening_and_filter_delayed();
    } else if (choice == "4") {
      /* Benchmark final addition with plaintext (encryption) */
      scenario_addition();
    } else if (choice == "5") {
      /* Benchmark final subtraction with ciphertext (decryption) */
      scenario_subtraction();
    } else if (choice == "6") {
      /* Benchmark complete encryption, single element */
      scenario_encrypt_elem();
    } else if (choice == "7") {
      /* Benchmark complete decryption, single element */
      scenario_decrypt_elem();
    } else if (choice == "8") {
      /* Benchmark complete encryption, full message */
      scenario_encrypt_message();
    } else if (choice == "9") {
      /* Benchmark complete decryption, full message */
      scenario_decrypt_message();
    } else if (choice == "testSBox") {
      /* Run a given S-Box to test it */
      scenario_test_sbox();
    } else if (choice == "genRndAES") {
      /* Fill a table with random values for faster subsequent lookups with AES */
      scenario_fill_rnd_table_aes();
    } else if (choice == "genRndChacha") {
      /* Fill a table with random values for faster subsequent lookups with Chacha */
      scenario_fill_rnd_table_chacha();
    } else if (choice == "setSecret") {
      /* Initialize the hash chain with a device secret for our countermeasure */
      scenario_set_device_secret();
    } else {
      print_format(mode, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
    }
    Serial.flush();
  }
}


/**
 * \brief          Arduino's setup loop run once at the beginning of the program. Setup the environment and initialize appropriate variables
 */
void setup() {
  /* Initialize serial and wait for port to open */
  Serial.begin(115200);
  while (!Serial)
    ;

  new_data = 0;

  srand(analogRead(A10));

  init_sboxes_4();
  init_sboxes_b4();

  /* Set the trigger PIN */
  pinMode(TriggerPQ, OUTPUT);

  print_format(-1, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
}

/**
 * \brief          Arduino's main loop run once continuously. The input is first fetched and then processed
 */
void loop() {
  recv_input();
  process_input();
}
