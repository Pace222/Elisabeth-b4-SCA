#include "filtering_4.h"

uint4_t S_BOXES_4[8][16];

/**
 * \brief          Initialize the `S_BOXES_4` table
 */
void init_sboxes_4() {
    uint8_t s_boxes_8_t[8][16] = {
        {0x03, 0x02, 0x06, 0x0C, 0x0A, 0x00, 0x01, 0x0B, 0x0D, 0x0E, 0x0A, 0x04, 0x06, 0x00, 0x0F, 0x05},
        {0x04, 0x0B, 0x04, 0x04, 0x04, 0x0F, 0x09, 0x0C, 0x0C, 0x05, 0x0C, 0x0C, 0x0C, 0x01, 0x07, 0x04},
        {0x0B, 0x0A, 0x0C, 0x02, 0x02, 0x0B, 0x0D, 0x0E, 0x05, 0x06, 0x04, 0x0E, 0x0E, 0x05, 0x03, 0x02},
        {0x05, 0x09, 0x0D, 0x02, 0x0B, 0x0A, 0x0C, 0x05, 0x0B, 0x07, 0x03, 0x0E, 0x05, 0x06, 0x04, 0x0B},
        {0x03, 0x00, 0x0B, 0x08, 0x0D, 0x0E, 0x0D, 0x0B, 0x0D, 0x00, 0x05, 0x08, 0x03, 0x02, 0x03, 0x05},
        {0x08, 0x0D, 0x0C, 0x0C, 0x03, 0x0F, 0x0C, 0x07, 0x08, 0x03, 0x04, 0x04, 0x0D, 0x01, 0x04, 0x09},
        {0X04, 0x02, 0x09, 0x0D, 0x0A, 0x0C, 0x0A, 0x07, 0x0C, 0x0E, 0x07, 0x03, 0x06, 0x04, 0x06, 0x09},
        {0X0A, 0x02, 0x05, 0x05, 0x03, 0x0D, 0x0F, 0x01, 0x06, 0x0E, 0x0B, 0x0B, 0x0D, 0x03, 0x01, 0x0F}
    };
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 16; j++) {
            S_BOXES_4[i][j] = uint4_new(s_boxes_8_t[i][j]);
        }
    }
}

/**
 * \brief          Elisabeth-4 filter function (one round)
 * \param[in]      block: The input block
 * \return         The filter function output
 */
uint4_t filter_block_4(const uint4_t* block) {
    uint4_t first_layer_output[BLOCK_WIDTH_4 - 1];

    for (int i = 0; i < BLOCK_WIDTH_4 - 1; i++) {
        first_layer_output[i] = S_BOXES_4[i][uint4_add(block[i], block[(i + 1) % (BLOCK_WIDTH_4 - 1)])];
    }

    uint4_t second_layer_output = uint4_new(0);
    for (int i = 0; i < BLOCK_WIDTH_4 - 1; i++) {
        uint4_t sboxes_sum = uint4_add(first_layer_output[(i + 1) % (BLOCK_WIDTH_4 - 1)], first_layer_output[(i + 2) % (BLOCK_WIDTH_4 - 1)]);
        second_layer_output = uint4_add(second_layer_output, S_BOXES_4[i + 4][uint4_add(block[i], sboxes_sum)]);
    }

    return uint4_add(second_layer_output, block[BLOCK_WIDTH_4 - 1]);
}

/**
 * \brief          Elisabeth-4 2-share arithmetic masked filter function (one round)
 * \note           Not implemented
 * \param[in]      block_shares: The masked input block
 * \return         The masked filter function output
 */
masked masked_filter_block_4(const masked* block_shares) {
    return 0;
}

/**
 * \brief          Elisabeth-4 shuffled filter function (one round)
 * \note           Not implemented
 * \param[in]      block: The input block
 * \return         The filter function output
 */
uint4_t shuffled_filter_block_4(const uint4_t* block) {
    return 0;
}

/**
 * \brief          Elisabeth-4 2-share arithmetic masked and shuffled filter function (one round)
 * \note           Not implemented
 * \param[in]      block_shares: The masked input block
 * \return         The masked filter function output
 */
masked masked_shuffled_filter_block_4(const masked* block_shares) {
    return 0;
}

/**
 * \brief          Elisabeth-4 delayed filter function (one round) according to our countermeasure
 * \note           Not implemented
 * \param[in]      block: The input block
 * \return         The filter function output
 */
uint4_t filter_block_4_delayed(const uint4_t* block) {
    return 0;
}
