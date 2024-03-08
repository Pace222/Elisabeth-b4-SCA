// Define Interrupt pin
#define interruptPIN A1

// Define trigger  PIN
#define TriggerPQ A0

void benchmark_whitening(uint4_t* keyround, uint4_t* key, rng* r, uint32_t* precomputed_random_values) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  random_whitened_subset(keyround, key, r, precomputed_random_values);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);
}

void benchmark_filter_block(uint4_t* filter_el, uint4_t* block, int mode) {
  uint4_t (*filter_block) (uint4_t*) = mode ? filter_block_4 : filter_block_b4;

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

void benchmark_whitening_and_filter(uint4_t* key_el, uint4_t* key, rng* r, uint32_t* precomputed_random_values) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  random_whitened_subset(keyround, key, r, precomputed_random_values);
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

void benchmark_encrypt_element(uint4_t* cipher_el, uint4_t plain_el, uint4_t* key, rng* r, uint32_t* precomputed_random_values) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];

  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  random_whitened_subset(keyround, key, r, precomputed_random_values);
  uint4_t res = uint4_add(plain_el, filter(keyround, r->mode));

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  *cipher_el = res;
}

void benchmark_decrypt_element(uint4_t* decrypted_el, uint4_t cipher_el, uint4_t* key, rng* r, uint32_t* precomputed_random_values) {
  uint4_t keyround[r->mode ? KEYROUND_WIDTH_4 : KEYROUND_WIDTH_B4];

  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  random_whitened_subset(keyround, key, r, precomputed_random_values);
  uint4_t res = uint4_add(cipher_el, uint4_neg(filter(keyround, r->mode)));

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);

  *decrypted_el = res;
}

void benchmark_encrypt_message(uint4_t* ciphertext, uint4_t* plaintext, uint4_t* key, rng* r, size_t length, uint32_t* precomputed_random_values) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  encrypt(ciphertext, plaintext, key, r, length, precomputed_random_values);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);
}

void benchmark_decrypt_message(uint4_t* decrypted, uint4_t* ciphertext, uint4_t* key, rng* r, size_t length, uint32_t* precomputed_random_values) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  decrypt(decrypted, ciphertext, key, r, length, precomputed_random_values);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);
}