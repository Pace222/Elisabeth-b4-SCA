#ifndef FILTERING_B4_H
#define FILTERING_B4_H

    #include "types.h"
    #include "masking.h"

static uint4_t S_BOXES_B4[18][16];

void init_sboxes_b4();
uint4_t filter_block_b4(const uint4_t*);
packed masked_filter_block_b4(const packed*);
uint4_t shuffled_filter_block_b4(const uint4_t*);
packed masked_shuffled_filter_block_b4(const packed*);
uint4_t filter_block_b4_delayed(const uint4_t*);

#endif