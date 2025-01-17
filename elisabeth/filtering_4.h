#ifndef FILTERING_4_H
#define FILTERING_4_H

#include "types.h"
#include "masking.h"

extern uint4_t S_BOXES_4[8][16];

void init_sboxes_4();
uint4_t filter_block_4(const uint4_t*);
masked masked_filter_block_4(const masked*);
uint4_t shuffled_filter_block_4(const uint4_t*);
masked masked_shuffled_filter_block_4(const masked*);
uint4_t filter_block_4_delayed(const uint4_t*);

#endif