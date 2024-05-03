#ifndef ENCRYPTION_H
#define ENCRYPTION_H

#include <stddef.h>

#include "keystream.h"

void encrypt(uint4_t*, const uint4_t*, const uint4_t*, const rng** r, size_t);
void masked_encrypt(add_packed*, const add_packed*, const add_packed*, const rng** r, size_t length);
void decrypt(uint4_t*, const uint4_t*, const uint4_t*, const rng** r, size_t);
void masked_decrypt(add_packed*, const add_packed*, const add_packed*, const rng** r, size_t length);

#endif