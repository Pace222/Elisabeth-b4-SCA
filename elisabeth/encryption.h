#ifndef ENCRYPTION_H
#define ENCRYPTION_H

#include <stddef.h>

#include "keystream.h"

void encrypt(uint4_t*, const uint4_t*, const uint4_t*, const rng* r, size_t, const uint32_t*);
void decrypt(uint4_t*, const uint4_t*, const uint4_t*, const rng* r, size_t, const uint32_t*);

#endif