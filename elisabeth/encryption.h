#ifndef ENCRYPTION_H
#define ENCRYPTION_H

#include <stddef.h>

#include "masking.h"
#include "keystream.h"

void encrypt(uint4_t*, const uint4_t*, const uint4_t*, const rng**, int);
void masked_encrypt(masked*, const masked*, const masked*, const rng**, int);
void decrypt(uint4_t*, const uint4_t*, const uint4_t*, const rng**, int);
void masked_decrypt(masked*, const masked*, const masked*, const rng**, int);

#endif