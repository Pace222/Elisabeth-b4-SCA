#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "encryption.h"

void fill_random_uint4_t(uint4_t* array, size_t length) {
  //srand(analogRead(A10));
  for (int i = 0; i < length; i++) {
    array[i] = uint4_new((uint8_t) rand());
  }
}

int hex2int(char ch) {
  if (ch >= '0' && ch <= '9')
      return ch - '0';
  if (ch >= 'A' && ch <= 'F')
      return ch - 'A' + 10;
  if (ch >= 'a' && ch <= 'f')
      return ch - 'a' + 10;
  return -1;
}

int main() {
  init_sboxes_4();
  srand(time(0));

  uint4_t plaintext[16];
  uint4_t ciphertext[16];
  uint4_t decrypted[16];
  uint4_t key[KEY_WIDTH_4];

  while (1) {
    uint8_t* seed = (uint8_t*) "\xA5\x60\x20\x88\x28\x35\xDF\xE2\xA8\x64\x7D\x9D\x7A\xFF\x38\x80";
    rng_aes r_enc, r_dec;
    rng_new_aes(&r_enc, seed, 1);
    rng_new_aes(&r_dec, seed, 1);
    const char* plain_str = "569A28F2D14026FA39BE674640B8A223BE744818E6812F2F1F1302B0AC9D95F7";
    for (int i = 0; i < strlen(plain_str); i++) {
      plaintext[i] = uint4_new(hex2int(plain_str[i]));
    }
    const char* key_str = "2EB2E1552681D58BA9611553F2DF956EB67A20F09156F0120B7EEAC550F314E3BBE41054DD16DC330877BEF3B8BA52A3753CDC0AD535376156979934FD2E99B2B0A1B515F52C2AADC1C96F14F8A224BAA1668277E9B4F41EE518D2A18F56043CF484DDD5CA9F2E30C366FDB949DF7DF622166E97DCAF6E5C18B36FE80C664865";
    for (int i = 0; i < strlen(key_str); i++) {
      key[i] = uint4_new(hex2int(key_str[i]));
    }

    encrypt(ciphertext, plaintext, key, &r_enc.r, 16);
    decrypt(decrypted, ciphertext, key, &r_dec.r, 16);

    printf("Seed (big-endian) : ");
    printf("%llX", seed);
    printf(" Plaintext : ");
    for (int i = 0; i < 16; i++) printf("%X", ((char*) plaintext)[i]);
    printf(" Key : ");
    for (int i = 0; i < KEY_WIDTH_4; i++) printf("%X", ((char*) key)[i]);
    printf(" Ciphertext : ");
    for (int i = 0; i < 16; i++) printf("%X", ((char*) ciphertext)[i]);
    printf(" Decrypted : ");
    for (int i = 0; i < 16; i++) printf("%X", ((char*) decrypted)[i]);
    printf("\n");
  }

  return 0;
}