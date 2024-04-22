#ifndef ENCRYPTION_H
#define ENCRYPTION_H

#include <stddef.h>

#include "keystream.h"

void encrypt(uint4_t*, const uint4_t*, const uint4_t*, const rng** r, size_t);
void protected_encrypt(uint4_t[][N_SHARES], const uint4_t[][N_SHARES], const uint4_t[][N_SHARES], const rng** r, size_t length);
void decrypt(uint4_t*, const uint4_t*, const uint4_t*, const rng** r, size_t);
void protected_decrypt(uint4_t[][N_SHARES], const uint4_t[][N_SHARES], const uint4_t[][N_SHARES], const rng** r, size_t length);

#endif