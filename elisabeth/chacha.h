// SOURCE: Hervé Pelletier - PQ_Kyber
#define CHACHA_KEYLEN 16

typedef struct
{
  uint32_t input[16]; /* could be compressed */
} ECRYPT_ctx;

void ECRYPT_keystream_bytes(ECRYPT_ctx *, uint8_t *, uint32_t);
void ECRYPT_init_ctx(ECRYPT_ctx *, const uint8_t *);