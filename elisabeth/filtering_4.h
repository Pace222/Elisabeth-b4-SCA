#ifndef FILTERING_4_H
#define FILTERING_4_H

#include "types.h"
#include "masking.h"

#define KEY_WIDTH_4 256
#define BLOCK_WIDTH_4 5
#define KEYROUND_WIDTH_4 60

static uint4_t S_BOXES_4[8][16];

void init_sboxes_4();
uint4_t filter_block_4(const uint4_t*);
packed protected_filter_block_4_mask_everything(const packed*);

#endif