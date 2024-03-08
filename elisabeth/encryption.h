#ifndef ENCRYPTION_H
#define ENCRYPTION_H

#include <stddef.h>

#include "keystream.h"

void encrypt(uint4_t*, uint4_t*, uint4_t*, rng* r, size_t, uint32_t*);
void decrypt(uint4_t*, uint4_t*, uint4_t*, rng* r, size_t, uint32_t*);

#endif