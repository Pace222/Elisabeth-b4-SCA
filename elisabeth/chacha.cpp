#include <stdint.h>
#include "chacha.h"

#define U8V(v) ((uint8_t) ((v) & 0XFF))
#define U32V(v) ((uint32_t) ((v) & 0xFFFFFFFF))

#define U8TO32_LITTLE(p) \
  (((uint32_t)((p)[0])      ) | \
   ((uint32_t)((p)[1]) <<  8) | \
   ((uint32_t)((p)[2]) << 16) | \
   ((uint32_t)((p)[3]) << 24))

#define ROTL32(v, n) \
    (U32V((v) << (n)) | ((v) >> (32 - (n))))

#define ROTATE(v,c) (ROTL32(v,c))
#define XOR(v,w) ((v) ^ (w))
#define PLUS(v,w) (U32V((v) + (w)))
#define PLUSONE(v) (PLUS((v),1))

#define QUARTERROUND(a,b,c,d) \
    x[a] = PLUS(x[a],x[b]); x[d] = ROTATE(XOR(x[d],x[a]),16); \
    x[c] = PLUS(x[c],x[d]); x[b] = ROTATE(XOR(x[b],x[c]),12); \
    x[a] = PLUS(x[a],x[b]); x[d] = ROTATE(XOR(x[d],x[a]), 8); \
    x[c] = PLUS(x[c],x[d]); x[b] = ROTATE(XOR(x[b],x[c]), 7);

static void salsa20_wordtobyte(uint8_t output[64], const uint32_t input[16])
{
    uint32_t x[16];
    int i;

    for (i = 0;i < 16;++i) x[i] = input[i];
    for (i = 8;i > 0;i -= 2) {
        QUARTERROUND( 0, 4, 8,12)
        QUARTERROUND( 1, 5, 9,13)
        QUARTERROUND( 2, 6,10,14)
        QUARTERROUND( 3, 7,11,15)
        QUARTERROUND( 0, 5,10,15)
        QUARTERROUND( 1, 6,11,12)
        QUARTERROUND( 2, 7, 8,13)
        QUARTERROUND( 3, 4, 9,14)
    }
    for (i = 0;i < 16;++i) x[i] = PLUS(x[i],input[i]);
    /*
     * there is absolutely no way of making the following line compile,
     * grow the fuck up and write semantically correct code
    for (i = 0;i < 16;++i) U32TO8_LITTLE(output + 4 * i,x[i]);
     */
    for (i = 0;i < 16;++i)
    {
        output[4*i + 0] = U8V(x[i] >> 0);
        output[4*i + 1] = U8V(x[i] >> 8);
        output[4*i + 2] = U8V(x[i] >> 16);
        output[4*i + 3] = U8V(x[i] >> 24);
    }
}

static char sigma[16];

static char tau[16];

void ECRYPT_keysetup(ECRYPT_ctx *x, const uint8_t *k, uint32_t kbits, uint32_t ivbits)
{
    const char *constants;
    sigma[0]='e';
    sigma[1]='x';
    sigma[2]='p';
    sigma[3]='a';
    sigma[4]='n';
    sigma[5]='d';
    sigma[6]=' ';
    sigma[7]='3';
    sigma[8]='2';
    sigma[9]='-';
    sigma[10]='b';
    sigma[11]='y';
    sigma[12]='t';
    sigma[13]='e';
    sigma[14]=' ';
    sigma[15]='k';

    tau[0]='e';
    tau[1]='x';
    tau[2]='p';
    tau[3]='a';
    tau[4]='n';
    tau[5]='d';
    tau[6]=' ';
    tau[7]='1';
    tau[8]='6';
    tau[9]='-';
    tau[10]='b';
    tau[11]='y';
    tau[12]='t';
    tau[13]='e';
    tau[14]=' ';
    tau[15]='k';


    x->input[4] = U8TO32_LITTLE(k + 0);
    x->input[5] = U8TO32_LITTLE(k + 4);
    x->input[6] = U8TO32_LITTLE(k + 8);
    x->input[7] = U8TO32_LITTLE(k + 12);
    if (kbits == 256) { /* recommended */
        k += 16;
        constants = sigma;
    } else { /* kbits == 128 */
        constants = tau;
    }
    x->input[8] = U8TO32_LITTLE(k + 0);
    x->input[9] = U8TO32_LITTLE(k + 4);
    x->input[10] = U8TO32_LITTLE(k + 8);
    x->input[11] = U8TO32_LITTLE(k + 12);
    x->input[0] = U8TO32_LITTLE(constants + 0);
    x->input[1] = U8TO32_LITTLE(constants + 4);
    x->input[2] = U8TO32_LITTLE(constants + 8);
    x->input[3] = U8TO32_LITTLE(constants + 12);
}

void ECRYPT_ivsetup(ECRYPT_ctx *x, const uint8_t *iv)
{
    x->input[12] = 0;
    x->input[13] = 0;
    x->input[14] = U8TO32_LITTLE(iv + 0);
    x->input[15] = U8TO32_LITTLE(iv + 4);
}

void ECRYPT_encrypt_bytes(ECRYPT_ctx *x, const uint8_t *m, uint8_t *c, uint32_t bytes)
{
    uint8_t output[64] = { 0x00 };
    int i = 0;

    if (!bytes) return;
    for (;;) {
        salsa20_wordtobyte(output,x->input);
        x->input[12] = PLUSONE(x->input[12]);
        if (!x->input[12]) {
            x->input[13] = PLUSONE(x->input[13]);
            /* stopping at 2^70 bytes per nonce is user's responsibility */
        }
        if (bytes <= 64) {
            for (i = 0;i < bytes;++i) c[i] = m[i] ^ output[i];
            return;
        }
        for (i = 0;i < 64;++i) c[i] = m[i] ^ output[i];
        bytes -= 64;
        c += 64;
        m += 64;
    }
}

void ECRYPT_keystream_bytes(ECRYPT_ctx *x, uint8_t *stream, uint32_t bytes)
{
    uint32_t i;
    for (i = 0; i < bytes; ++i) stream[i] = 0;
    ECRYPT_encrypt_bytes(x, stream, stream, bytes);
}

void ECRYPT_init_ctx(ECRYPT_ctx *x, const uint8_t *key) {
    int i = 0;
    uint8_t iv[8];

    for (i = 0; i < 8; ++i) {
        iv[i] = 0x00;
    }

    ECRYPT_keysetup(x, key, 8 * CHACHA_KEYLEN, 0);
    ECRYPT_ivsetup(x, iv);
}