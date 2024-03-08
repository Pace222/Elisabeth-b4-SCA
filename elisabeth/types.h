#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

typedef uint8_t uint4_t;

#define uint4_new(val) ((val) & 0x0F)
#define uint4_add(a, b) (((a) + (b)) & 0x0F)
#define uint4_neg(a) ((0x10 - (a)) & 0x0F)

#endif