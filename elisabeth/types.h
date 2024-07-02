#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

#define KEY_WIDTH_4 256
#define BLOCK_WIDTH_4 5
#define KEYROUND_WIDTH_4 60

#define KEY_WIDTH_B4 512
#define BLOCK_WIDTH_B4 7
#define KEYROUND_WIDTH_B4 98

typedef uint8_t uint4_t;

#define uint4_new(val) ((val) & 0x0F)
#define uint4_add(a, b) (((a) + (b)) & 0x0F)
#define uint4_neg(a) ((0x10 - (a)) & 0x0F)

#endif