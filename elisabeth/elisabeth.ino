
extern "C" {
  #include "encryption.h"
  #include "generator_aes.h"
}

// Define Interrupt pin
#define interruptPIN A1

// Define trigger  PIN
#define TriggerPQ A0

void benchmark_whitening(uint4_t* keyround, uint4_t* key, rng* r) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  random_whitened_subset(keyround, key, r);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);
}

void benchmark_filter_block(uint4_t* filter_el, uint4_t* block, int mode) {
  uint4_t (*filter_block) (const uint4_t*) = mode ? filter_block_4 : filter_block_b4;

  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  uint4_t res = filter_block(block);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  *filter_el = res;
}

void benchmark_filter(uint4_t* key_el, uint4_t* keyround, int mode) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  uint4_t res = filter(keyround, mode);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  *key_el = res;
}

void benchmark_whitening_and_filter(uint4_t* key_el, uint4_t* key, rng* r) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  random_whitened_subset(keyround, key, r);
  uint4_t res = filter(keyround, r->mode);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  *key_el = res;
}

void benchmark_addition(uint4_t* cipher_el, uint4_t plain_el, uint4_t key_el) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  uint4_t r = uint4_add(plain_el, key_el);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  *cipher_el = r;
}

void benchmark_subtraction(uint4_t* decrypted_el, uint4_t cipher_el, uint4_t key_el) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  uint4_t r = uint4_add(cipher_el, uint4_neg(key_el));

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  *decrypted_el = r;
}

void benchmark_encrypt_element(uint4_t* cipher_el, uint4_t plain_el, uint4_t* key, rng* r) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];

  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  random_whitened_subset(keyround, key, r);
  uint4_t res = uint4_add(plain_el, filter(keyround, r->mode));

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  *cipher_el = res;
}

void benchmark_decrypt_element(uint4_t* decrypted_el, uint4_t cipher_el, uint4_t* key, rng* r) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];

  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  random_whitened_subset(keyround, key, r);
  uint4_t res = uint4_add(cipher_el, uint4_neg(filter(keyround, r->mode)));

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  *decrypted_el = res;
}

void benchmark_encrypt_message(uint4_t* ciphertext, uint4_t* plaintext, uint4_t* key, const rng** r, size_t length) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  encrypt(ciphertext, plaintext, key, r, length);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);
}

void benchmark_decrypt_message(uint4_t* decrypted, uint4_t* ciphertext, uint4_t* key, const rng** r, size_t length) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  decrypt(decrypted, ciphertext, key, r, length);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);
}

#define MAX_MESSAGE_SIZE 64

#define MAX_INPUT_SIZE 10 + AES_KEYLEN + MAX_MESSAGE_SIZE + KEY_WIDTH_B4
#define START '<'
#define DELIMITER ','
#define STOP '>'

int new_data;
char input_buffer[MAX_INPUT_SIZE + 1];
size_t process_head;

int hex2int(char ch) {
  if (ch >= '0' && ch <= '9')
      return ch - '0';
  if (ch >= 'A' && ch <= 'F')
      return ch - 'A' + 10;
  if (ch >= 'a' && ch <= 'f')
      return ch - 'a' + 10;
  return -1;
}

void fill_random_uint4_t(uint4_t* array, size_t length) {
  for (int i = 0; i < length; i++) {
    array[i] = uint4_new((uint8_t) rand());
  }
}

String read_until(char delimiter) {
  char* start = input_buffer + process_head;
  char c = input_buffer[process_head++];
  while (c != '\0' && c != delimiter) {
    c = input_buffer[process_head++];
  }

  if (c == delimiter) {
    input_buffer[process_head - 1] = '\0';
    return String(start);
  } else {
    process_head -= 1;
    return String();
  }
}

int fill_array_from_user_hex_bytes(uint8_t* array, size_t length, char expected_delimiter) {
  char c1, c2;
  int n1, n2;

  // Skip first nibble if array expects an odd number of nibbles.
  if (length % 2 == 1) {
    c1 = input_buffer[process_head];
    if (c1 == '\0' || hex2int(c1) < 0) {
      return 0;
    } else {
      process_head += 1;
    }
  }

  for (int i = 0; i < length; i++) {
    c1 = input_buffer[process_head];
    c2 = input_buffer[process_head + 1];
    if (c1 == '\0' || c2 == '\0') {
      return i;
    } else {
      n1 = hex2int(c1);
      n2 = hex2int(c2);
      if (n1 < 0 || n2 < 0) {
        return i;
      }

      array[i] = (n1 << 4) + n2;

      process_head += 2;
    }
  }

  c1 = input_buffer[process_head];
  if (c1 == '\0') {
    return length;
  }
  if (c1 != expected_delimiter) {
    return -1;
  }
  process_head += 1;
  return length;
}

int fill_array_from_user_hex(uint4_t* array, size_t length, char expected_delimiter) {
  char c;
  int n;

  // Skip first nibble if array expects an odd number of nibbles.
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
      return i;
    } else {
      n = hex2int(c);
      if (n < 0) {
        return i;
      }

      array[i] = uint4_new(n);

      process_head += 1;
    }
  }

  c = input_buffer[process_head];
  if (c == '\0') {
    return length;
  }
  if (c != expected_delimiter) {
    return -1;
  }
  process_head += 1;
  return length;
}

int fill_array_from_user_until(uint4_t* array, char delimiter) {
  char c;
  int n;
  int length;

  for (length = 0; length < MAX_MESSAGE_SIZE; length++) {
    c = input_buffer[process_head];
    if (c == delimiter) {
      process_head += 1;
      return length;
    } else if (c == '\0') {
      return length;
    } else {
      n = hex2int(c);
      if (n < 0) {
        return -1;
      }

      array[length] = uint4_new(n);

      process_head += 1;
    }
  }

  c = input_buffer[process_head];
  if (c == delimiter) {
    process_head += 1;
    return length;
  } else if (c == '\0') {
    return length;
  }
  
  return -1;
}

void print_format(int mode, String choice, String arguments, String description) {
  String str_to_print = "Format: <";

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

  str_to_print += arguments + "> | Elisabeth-4/b4 | Number of repetitions | Scenario | " + description;

  str_to_print += " All arguments must be in hexadecimal format, except for the number of repetitions which is in decimal.";

  Serial.println(str_to_print);

  if (choice == "\0") {
    Serial.println("  0: Benchmark whitening");
    Serial.println("  1: Benchmark single block of filter function");
    Serial.println("  2: Benchmark full filter function");
    Serial.println("  3: Benchmark whitening + full filter function");
    Serial.println("  4: Benchmark final addition with plaintext (encryption)");
    Serial.println("  5: Benchmark final subtraction with ciphertext (decryption)");
    Serial.println("  6: Benchmark complete encryption, single element");
    Serial.println("  7: Benchmark complete decryption, single element");
    Serial.println("  8: Benchmark complete encryption, full message");
    Serial.println("  9: Benchmark complete decryption, full message");
  }
}

void recv_input() {
  static int recv_in_progress = 0;
  static int truncated = 0;
  static size_t head = 0;
  char c;

  while (Serial.available() > 0 && !new_data) {
    c = Serial.read();

    if (recv_in_progress) {
      if (c != STOP) {
        input_buffer[head] = c;
        head++;
        if (head > MAX_INPUT_SIZE) {
          head = MAX_INPUT_SIZE;
          truncated = 1;
        }
      } else {
        input_buffer[head] = '\0'; // terminate the string
        recv_in_progress = 0;
        head = 0;
        if (truncated) {
          Serial.println("Input too long.");
          print_format(-1, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
          truncated = 0;
        } else {
          new_data = 1;
        }
      }
    } else if (c == START) {
      recv_in_progress = 1;
    }
  }
}

#include <Profiler.h>


uint8_t buf_seed_1[AES_KEYLEN], buf_seed_2[AES_KEYLEN];
uint4_t buf_message[MAX_MESSAGE_SIZE], buf_out[MAX_MESSAGE_SIZE];
uint4_t buf_arg[KEY_WIDTH_B4];
rng_aes rng_aes_list[MAX_MESSAGE_SIZE];
const rng* rng_list[MAX_MESSAGE_SIZE];

int BLOCK_WIDTH, KEYROUND_WIDTH, KEY_WIDTH;

String mode_str, repeat_str, choice;
int mode, repeat;
size_t actual_message_length;

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

int setup_repeat() {
    repeat_str = read_until(DELIMITER);
    repeat = repeat_str.toInt();
    if (repeat <= 0) {
      print_format(mode, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
      return 1;
    }
    
    return 0;
}

int setup_choice() {
    choice = read_until(DELIMITER);
    if (choice.length() == 0) {
      print_format(mode, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
      return 1;
    }
    
    return 0;
}

void scenario_whitening_seed() {
    // Format: [0-9A-Fa-f]. arg1 is the key. Output is the round key. Expects to have filled the random table in a previous command.
    if (fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the round key (SIZE: " + String(KEYROUND_WIDTH) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_whitening(buf_out, buf_arg, &rng_aes_list[0].r);

      for (int i = 0; i < KEYROUND_WIDTH; i++) Serial.print(((char*) buf_out)[i], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_filter_block() {
    // Format: [0-9A-Fa-f]. arg1 is a round key block. Output is the output of a single block of the filtering function.
    if (fill_array_from_user_hex(buf_arg, BLOCK_WIDTH, DELIMITER) != BLOCK_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is a round key block (SIZE: 1 + " + String(BLOCK_WIDTH) + " nibbles). Output is the output of a single block of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles).");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_filter_block(buf_out, buf_arg, mode);

      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_filter() {
    // Format: [0-9A-Fa-f]. arg1 is the round key. Output is the output of the filtering function.
    if (fill_array_from_user_hex(buf_arg, KEYROUND_WIDTH, DELIMITER) != KEYROUND_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the round key (SIZE: " + String(KEYROUND_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles).");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_filter(buf_out, buf_arg, mode);

      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_whitening_and_filter() {
    // Format: [0-9A-Fa-f]. arg1 is the key. Output is the output of the filtering function. Expects to have filled the random table in a previous command.
    if (fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the output of the filtering function (SIZE: " + String(sizeof(uint4_t)) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_whitening_and_filter(buf_out, buf_arg, &rng_aes_list[0].r);

      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_addition() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is a single element of the plaintext, arg2 is the output of the filtering function. Output is a single element of the ciphertext.
    if (fill_array_from_user_hex(buf_message, 1, DELIMITER) != 1 || fill_array_from_user_hex(buf_arg, 1, DELIMITER) != 1) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the plaintext (SIZE: 1 + " + String(1) + " nibbles), arg2 is the output of the filtering function (SIZE: 1 + " + String(1) + " nibbles). Output is a single element of the ciphertext (SIZE: " + String(1) + " nibbles).");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_addition(buf_out, buf_message[0], buf_arg[0]);

      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_subtraction() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is a single element of the ciphertext, arg2 is the output of the filtering function. Output is a single element of the plaintext.
    if (fill_array_from_user_hex(buf_message, 1, DELIMITER) != 1 || fill_array_from_user_hex(buf_arg, 1, DELIMITER) != 1) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the ciphertext (SIZE: 1 + " + String(1) + " nibbles), arg2 is the output of the filtering function (SIZE: 1 + " + String(1) + " nibbles). Output is a single element of the plaintext (SIZE: " + String(1) + " nibbles).");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_subtraction(buf_out, buf_message[0], buf_arg[0]);

      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_encrypt_elem() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is a single element of the plaintext, arg2 is the key. Output is a single element of the ciphertext. Expects to have filled the random table in a previous command.
    if (fill_array_from_user_hex(buf_message, 1, DELIMITER) != 1 || fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the plaintext (SIZE: 1 + " + String(1) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is a single element of the ciphertext (SIZE: " + String(1) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_encrypt_element(buf_out, buf_message[0], buf_arg, &rng_aes_list[0].r);

      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_decrypt_elem() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is a single element of the ciphertext, arg2 is the key. Output is a single element of the plaintext. Expects to have filled the random table in a previous command.
    if (fill_array_from_user_hex(buf_message, 1, DELIMITER) != 1 || fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is a single element of the ciphertext (SIZE: 1 + " + String(1) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is a single element of the plaintext (SIZE: " + String(1) + " nibbles). Expects to have filled the random table in a previous command.");
        return;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_decrypt_element(buf_out, buf_message[0], buf_arg, &rng_aes_list[0].r);

      Serial.print(buf_out[0], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_encrypt_message() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is the complete plaintext (message) to encrypt, arg2 is the key. Output is the complete ciphertext (encrypted message). Expects to have filled the random table in a previous command.
    if ((actual_message_length = fill_array_from_user_until(buf_message, DELIMITER)) <= 0 || fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the complete plaintext (message) to encrypt (SIZE: max " + String(MAX_MESSAGE_SIZE) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the complete ciphertext (encrypted message) (SIZE: same as plaintext). Expects to have filled the random table in a previous command.");
        return;
    }

    rng_list[0] = &rng_aes_list[0].r;
    for (int i = 1; i < actual_message_length; i++) {
      rng_aes_list[i - 1].r.copy(&rng_aes_list[i].r, &rng_aes_list[i - 1].r);
      rng_aes_list[i].r.next_elem(&rng_aes_list[i].r);
      rng_list[i] = &rng_aes_list[i].r;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_encrypt_message(buf_out, buf_message, buf_arg, rng_list, actual_message_length);

      for (int i = 0; i < actual_message_length; i++) Serial.print(((char*) buf_out)[i], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_decrypt_message() {
    // Format: [0-9A-Fa-f],[0-9A-Fa-f]. arg1 is the complete ciphertext (encrypted message) to decrypt, arg2 is the key. Output is the complete plaintext (decrypted message). Expects to have filled the random table in a previous command.
    if ((actual_message_length = fill_array_from_user_until(buf_message, DELIMITER)) <= 0 || fill_array_from_user_hex(buf_arg, KEY_WIDTH, DELIMITER) != KEY_WIDTH) {
        print_format(mode, choice, "[0-9A-Fa-f],[0-9A-Fa-f]", "arg1 is the complete ciphertext (encrypted message) to decrypt (SIZE: max " + String(MAX_MESSAGE_SIZE) + " nibbles), arg2 is the key (SIZE: " + String(KEY_WIDTH) + " nibbles). Output is the complete plaintext (decrypted message) (SIZE: same as ciphertext). Expects to have filled the random table in a previous command.");
        return;
    }

    rng_list[0] = &rng_aes_list[0].r;
    for (int i = 1; i < actual_message_length; i++) {
      rng_aes_list[i - 1].r.copy(&rng_aes_list[i].r, &rng_aes_list[i - 1].r);
      rng_aes_list[i].r.next_elem(&rng_aes_list[i].r);
      rng_list[i] = &rng_aes_list[i].r;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_decrypt_message(buf_out, buf_message, buf_arg, rng_list, actual_message_length);

      for (int i = 0; i < actual_message_length; i++) Serial.print(((char*) buf_out)[i], HEX);
      if (i < repeat - 1) {
        Serial.print(DELIMITER);
      }
    }
}

void scenario_fill_rnd_table_aes() {
  // Format: [0-9A-Fa-f]. arg1 is the seed for the PRNG (IV). No output.
  if (fill_array_from_user_hex_bytes(buf_seed_1, AES_KEYLEN, DELIMITER) != AES_KEYLEN) {
      print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(2 * AES_KEYLEN) + " nibbles). No output.");
      return;
  }

  switch_endianness(buf_seed_2, buf_seed_1, AES_KEYLEN);
  rng_new_aes(&rng_aes_list[0], buf_seed_2, mode);
  Serial.print("1");
}

void scenario_fill_rnd_table_chacha() {
  // Format: [0-9A-Fa-f]. arg1 is the seed for the PRNG (IV). No output.
  if (fill_array_from_user_hex_bytes(buf_seed_1, AES_KEYLEN, DELIMITER) != AES_KEYLEN) {
      print_format(mode, choice, "[0-9A-Fa-f]", "arg1 is the seed for the PRNG (IV) (SIZE: " + String(2 * AES_KEYLEN) + " nibbles). No output.");
      return;
  }

  switch_endianness(buf_seed_2, buf_seed_1, AES_KEYLEN);
  //rng_new_chacha(&rng_aes_list[0], buf_seed_2, mode);
  Serial.print("0");
}

void setup() {
  // Initialize serial and wait for port to open.
  Serial.begin(115200);
  while (!Serial)
    ;

  new_data = 0;

  init_sboxes_4();
  init_sboxes_b4();

  // Set the trigger PIN
  pinMode(TriggerPQ, OUTPUT);

  print_format(-1, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
}


void loop() {
  recv_input();
  process_input();
}

void process_input() {
  if (new_data) {
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
      // Benchmark whitening
      scenario_whitening_seed();
    } else if (choice == "1") {
      // Benchmark single block of filter function
      scenario_filter_block();
    } else if (choice == "2") {
      // Benchmark full filter function
      scenario_filter();
    } else if (choice == "3") {
      // Benchmark whitening + full filter function
      scenario_whitening_and_filter();
    } else if (choice == "4") {
      // Benchmark final addition with plaintext (encryption)
      scenario_addition();
    } else if (choice == "5") {
      // Benchmark final subtraction with ciphertext (decryption)
      scenario_subtraction();
    } else if (choice == "6") {
      // Benchmark complete encryption, single element
      scenario_encrypt_elem();
    } else if (choice == "7") {
      // Benchmark complete decryption, single element
      scenario_decrypt_elem();
    } else if (choice == "8") {
      // Benchmark complete encryption, full message
      scenario_encrypt_message();
    } else if (choice == "9") {
      // Benchmark complete decryption, full message
      scenario_decrypt_message();
    } else if (choice == "genRndAES") {
      // Fill a table with random values for faster subsequent lookups with AES.
      scenario_fill_rnd_table_aes();
    } else if (choice == "genRndArduino") {
      // Fill a table with random values for faster subsequent lookups with the built-in random function.
      scenario_fill_rnd_table_chacha();
    } else {
      print_format(mode, "\0", "arg1,arg2,arg3,...", "Arguments depend on the benchmark.");
    }
  }
}
