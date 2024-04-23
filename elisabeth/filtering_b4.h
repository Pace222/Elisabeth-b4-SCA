#ifndef FILTERING_B4_H
#define FILTERING_B4_H

#include "types.h"
#include "masking.h"

#define KEY_WIDTH_B4 512
#define BLOCK_WIDTH_B4 7
#define KEYROUND_WIDTH_B4 98

static uint4_t S_BOXES_B4[18][16];

void init_sboxes_b4();
uint4_t filter_block_b4(const uint4_t*);
packed protected_filter_block_b4_mask_everything(const packed*);

#endif