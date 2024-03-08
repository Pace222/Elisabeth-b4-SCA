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
    uint8_t* seed = (uint8_t*) 0x386F8B02;
    rng r_enc, r_dec;
    rng_new(&r_enc, seed, 1);
    rng_new(&r_dec, seed, 1);
    const char* plain_str = "A3F9F7DEDDBEE82D";
    for (int i = 0; i < strlen(plain_str); i++) {
      plaintext[i] = uint4_new(hex2int(plain_str[i]));
    }
    const char* key_str = "4529C2464C198BBA62C60AF845E400B596CA981348A7B9645E2DB3506245F1FDD084F85AEC8342D24AB857F48B2F1FC0A0E5242E4AF16397F8FE5ADD37D8554CBA03E26DEFB23A89F70BD6AE822F43BF9339D72E1D986BE042E40CC09356AA66D658D43D3EEE580F8B9D7413B0D4AC4C5C35382C3E8EA757B0EF78DAF98F7A56>";
    for (int i = 0; i < strlen(key_str); i++) {
      key[i] = uint4_new(hex2int(key_str[i]));
    }

    encrypt(ciphertext, plaintext, key, &r_enc, 16);
    decrypt(decrypted, ciphertext, key, &r_dec, 16);

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