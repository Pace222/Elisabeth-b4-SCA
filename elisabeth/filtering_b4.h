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
void protected_filter_block_b4_mask_everything(uint4_t*, const uint4_t[][N_SHARES]);
uint4_t protected_filter_block_b4_mask_only_input(const uint4_t[][N_SHARES]);

#endif