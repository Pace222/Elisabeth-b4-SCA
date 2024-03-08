#include <stddef.h>
#include <stdlib.h>
#include <string.h>

extern "C"
{
#include "encryption.h"
}


// Define trigger  PIN
#define TriggerPQ A0

#define PLAINTEXT_SIZE 16
#define KEY_WIDTH KEY_WIDTH_4
#define KEYROUND_WIDTH KEYROUND_WIDTH_4
#define ROUNDS 4


size_t seed;
rng r_enc, r_dec;

uint4_t plaintext[PLAINTEXT_SIZE];

uint4_t keyround[KEYROUND_WIDTH];
uint4_t key[KEY_WIDTH];

uint4_t ciphertext[PLAINTEXT_SIZE];

uint4_t decrypted[PLAINTEXT_SIZE];


void setup() {
  // Initialize serial and wait for port to open.
  Serial.begin(115200);
  while (!Serial)
    ;

  init_sboxes_4();

  // Set the trigger PIN
  Serial.println("TriggerStart");
  pinMode(TriggerPQ, OUTPUT);
}

// Write me a function that accepts as argument a function and a list of arguments. The function simply calls the function with the given arguments.
// The function should accept arguments and call f with those arguments.
void benchmark_function(void (*f)(uint4_t*, uint4_t*, uint4_t*, rng*, size_t), uint4_t* arg1, uint4_t* arg2, uint4_t* arg3, rng* arg4, size_t arg5) {
  noInterrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);  

  f(arg1, arg2, arg3, arg4, arg5);

  interrupts();
  //Run the trigger Low to High
  digitalWrite(TriggerPQ, LOW);
  delayMicroseconds(100);
  digitalWrite(TriggerPQ, HIGH);
}

void fill_random_uint4_t(uint4_t* array, size_t length) {
  srand(analogRead(A10));
  for (int i = 0; i < length; i++) {
    array[i] = uint4_new((uint8_t) rand());
  }
}


void loop() {
  // put your main code here, to run repeatedly:

  // Debug encryption + decryption
  {
    seed = rand();

    rng_new(&r_enc, seed, 1);
    rng_new(&r_dec, seed, 1);
    fill_random_uint4_t(plaintext, PLAINTEXT_SIZE);
    fill_random_uint4_t(key, KEY_WIDTH);

    encrypt(ciphertext, plaintext, key, &r_enc, PLAINTEXT_SIZE);
    decrypt(decrypted, ciphertext, key, &r_dec, PLAINTEXT_SIZE);

    Serial.print("Seed : ");
    Serial.println(seed);

    Serial.print("Plaintext : ");
    for (int i = 0; i < PLAINTEXT_SIZE; i++) Serial.print(((char*) plaintext)[i], HEX);
    Serial.println();

    Serial.print("Key : ");
    for (int i = 0; i < KEY_WIDTH; i++) Serial.print(((char*) key)[i], HEX);
    Serial.println();

    Serial.print("Ciphertext: ");
    for (int i = 0; i < PLAINTEXT_SIZE; i++) Serial.print(((char*) ciphertext)[i], HEX);
    Serial.println();

    Serial.print("Decrypted : ");
    for (int i = 0; i < PLAINTEXT_SIZE; i++) Serial.print(((char*) decrypted)[i], HEX);
    Serial.println();

    encrypt(plaintext, ciphertext, key, &r_enc, PLAINTEXT_SIZE);
    decrypt(decrypted, plaintext, key, &r_dec, PLAINTEXT_SIZE);

    Serial.print("Ciphertext 2: ");
    for (int i = 0; i < PLAINTEXT_SIZE; i++) Serial.print(((char*) plaintext)[i], HEX);
    Serial.println();

    Serial.print("Decrypted : ");
    for (int i = 0; i < PLAINTEXT_SIZE; i++) Serial.print(((char*) decrypted)[i], HEX);
    Serial.println();

    encrypt(ciphertext, plaintext, key, &r_enc, PLAINTEXT_SIZE);
    decrypt(decrypted, ciphertext, key, &r_dec, PLAINTEXT_SIZE);

    Serial.print("Ciphertext 3: ");
    for (int i = 0; i < PLAINTEXT_SIZE; i++) Serial.print(((char*) ciphertext)[i], HEX);
    Serial.println();

    Serial.print("Decrypted : ");
    for (int i = 0; i < PLAINTEXT_SIZE; i++) Serial.print(((char*) decrypted)[i], HEX);
    Serial.println();

    fill_random_uint4_t(plaintext, PLAINTEXT_SIZE);
    fill_random_uint4_t(key, KEY_WIDTH);

    Serial.println();
  }

  // Debug whitening
  {
    seed = rand();
    rng_new(&r_enc, seed, 1);
    fill_random_uint4_t(key, KEY_WIDTH);
    random_whitened_subset(keyround, key, &r_enc);

    Serial.print("Seed : ");
    Serial.println(seed, HEX);
    Serial.print("Key : ");
    for (int i = 0; i < KEY_WIDTH; i++) Serial.print(((char*) key)[i], HEX);
    Serial.println();
    Serial.print("Keyround : ");
    for (int i = 0; i < KEYROUND_WIDTH; i++) Serial.print(((char*) keyround)[i], HEX);
    Serial.println();
  }

  // Debug full filter function
  {
    fill_random_uint4_t(keyround, KEYROUND_WIDTH);
    size_t res = filter(keyround, 1);

    Serial.print("Input : ");
    for (int i = 0; i < KEYROUND_WIDTH; i++) Serial.print(((char*) keyround)[i], HEX);
    Serial.println();
    Serial.print("Output : ");
    Serial.println(*((char*) &res), HEX);
  }
}
