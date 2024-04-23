#include <stddef.h>

#include "filtering_b4.h"
#include "masking.h"

void init_sboxes_b4() {
    uint8_t s_boxes_8_t[18][16] = {
        {0x0A, 0x06, 0x0B, 0x08, 0x04, 0x09, 0x08, 0x0C, 0x06, 0x0A, 0x05, 0x08, 0x0C, 0x07, 0x08, 0x04},
        {0x09, 0x01, 0x05, 0x05, 0x00, 0x0C, 0x02, 0x06, 0x07, 0x0F, 0x0B, 0x0B, 0x10, 0x04, 0x0E, 0x0A},
        {0x0D, 0x0E, 0x0E, 0x02, 0x03, 0x09, 0x03, 0x05, 0x03, 0x02, 0x02, 0x0E, 0x0D, 0x07, 0x0D, 0x0B},
        {0x02, 0x09, 0x08, 0x0B, 0x0D, 0x08, 0x01, 0x07, 0x0E, 0x07, 0x08, 0x05, 0x03, 0x08, 0x0F, 0x09},
        {0x0B, 0x03, 0x0F, 0x09, 0x00, 0x00, 0x0C, 0x00, 0x05, 0x0D, 0x01, 0x07, 0x10, 0x10, 0x04, 0x10},
        {0x0F, 0x0C, 0x01, 0x0F, 0x0E, 0x01, 0x06, 0x0C, 0x01, 0x04, 0x0F, 0x01, 0x02, 0x0F, 0x0A, 0x04},
        {0x06, 0x0E, 0x0D, 0x00, 0x07, 0x0E, 0x0C, 0x03, 0x0A, 0x02, 0x03, 0x10, 0x09, 0x02, 0x04, 0x0D},
        {0x0C, 0x00, 0x04, 0x01, 0x0F, 0x0B, 0x04, 0x00, 0x04, 0x10, 0x0C, 0x0F, 0x01, 0x05, 0x0C, 0x10},
        {0x0B, 0x00, 0x0F, 0x0A, 0x09, 0x0B, 0x09, 0x02, 0x05, 0x10, 0x01, 0x06, 0x07, 0x05, 0x07, 0x0E},
        {0x0D, 0x03, 0x0B, 0x0B, 0x08, 0x09, 0x08, 0x0C, 0x03, 0x0D, 0x05, 0x05, 0x08, 0x07, 0x08, 0x04},
        {0x0A, 0x02, 0x08, 0x04, 0x0F, 0x0B, 0x06, 0x04, 0x06, 0x0E, 0x08, 0x0C, 0x01, 0x05, 0x0A, 0x0C},
        {0x0D, 0x08, 0x0E, 0x08, 0x02, 0x05, 0x03, 0x0B, 0x03, 0x08, 0x02, 0x08, 0x0E, 0x0B, 0x0D, 0x05},
        {0x0D, 0x0F, 0x02, 0x05, 0x05, 0x0F, 0x09, 0x0B, 0x03, 0x01, 0x0E, 0x0B, 0x0B, 0x01, 0x07, 0x05},
        {0x0D, 0x00, 0x0A, 0x0A, 0x06, 0x07, 0x03, 0x0E, 0x03, 0x10, 0x06, 0x06, 0x0A, 0x09, 0x0D, 0x02},
        {0x00, 0x04, 0x07, 0x00, 0x09, 0x04, 0x0C, 0x00, 0x10, 0x0C, 0x09, 0x10, 0x07, 0x0C, 0x04, 0x10},
        {0x04, 0x0B, 0x06, 0x03, 0x0F, 0x06, 0x0C, 0x02, 0x0C, 0x05, 0x0A, 0x0D, 0x01, 0x0A, 0x04, 0x0E},
        {0x03, 0x0C, 0x01, 0x08, 0x08, 0x0F, 0x0D, 0x0F, 0x0D, 0x04, 0x0F, 0x08, 0x08, 0x01, 0x03, 0x01},
        {0x0B, 0x03, 0x02, 0x0C, 0x03, 0x08, 0x04, 0x02, 0x05, 0x0D, 0x0E, 0x04, 0x0D, 0x08, 0x0C, 0x0E}
    };
    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < 16; j++) {
            S_BOXES_B4[i][j] = uint4_new(s_boxes_8_t[i][j]);
        }
    }
}

uint4_t filter_block_b4(const uint4_t* block) {
    size_t new_width = BLOCK_WIDTH_B4 - 1;
    uint4_t x[BLOCK_WIDTH_B4];
    for (int i = 0; i < BLOCK_WIDTH_B4; i++) {
        x[i] = block[i];
    }
    uint4_t y[new_width];
    uint4_t z[new_width];
    uint4_t t[new_width];
    uint4_t res;

    for (int i = 0; i < new_width / 2; i++) {
        x[2*i + 1] = uint4_add(x[2*i + 1], x[2*i]);
    }

    for (int i = 0; i < new_width; i++) {
        y[i] = S_BOXES_B4[i][x[i]];
    }

    for (int i = 0; i < new_width / 2; i++) {
        z[2*i] = uint4_add(y[(2*i + 5) % new_width], y[2*i]);
        z[2*i + 1] = uint4_add(y[(2*i + 4) % new_width], y[2*i + 1]);
    }

    for (int i = 0; i < new_width; i++) {
        z[i] = uint4_add(z[i], x[(i + 2) % new_width]);
        z[i] = S_BOXES_B4[i + new_width][z[i]];
    }

    for (int i = 0; i < new_width / 3; i++) {
        t[3*i] = uint4_add(uint4_add(z[3*i], z[3*i + 1]), z[3*i + 2]);
        t[3*i + 1] = uint4_add(z[3*i + 1], z[(3*i + 3) % new_width]);
        t[3*i + 2] = uint4_add(uint4_add(z[3*i + 2], z[(3*i + 3) % new_width]), y[3*i]);
    }

    t[0] = uint4_add(t[0], x[5]);
    t[1] = uint4_add(t[1], x[4]);
    t[2] = uint4_add(t[2], x[3]);
    t[3] = uint4_add(t[3], x[1]);
    t[4] = uint4_add(t[4], x[0]);
    t[5] = uint4_add(t[5], x[2]);

    res = x[new_width];
    for (int i = 0; i < new_width; i++) {
        res = uint4_add(res, S_BOXES_B4[i + 2*new_width][t[i]]);
    }
    return res;
}

packed protected_filter_block_b4_mask_everything(const packed* block_shares) {
    // Protection under second order DPA
    size_t new_width = BLOCK_WIDTH_B4 - 1;

    packed x_shares[BLOCK_WIDTH_B4];
    for (int i = 0; i < BLOCK_WIDTH_B4; i++) {                                                                                      // for (int i = 0; i < BLOCK_WIDTH_B4; i++) {
        x_shares[i] = block_shares[i];                                                                                              //     x[i] = block[i];
    }                                                                                                                               // }

    packed y_shares[new_width];
    packed z_shares[new_width];
    packed t_shares[new_width];

    for (int i = 0; i < new_width / 2; i++) {
        x_shares[2*i + 1] = masked_addition(x_shares[2*i + 1], x_shares[2*i]);                                                      // x[2*i + 1] = uint4_add(x[2*i + 1], x[2*i]);
    }

    for (int i = 0; i < new_width; i++) {
        y_shares[i] = masked_sbox_second_order(x_shares[i], S_BOXES_B4[i]);                                                         // y[i] = S_BOXES_B4[i][x[i]];
    }

    for (int i = 0; i < new_width / 2; i++) {
        z_shares[2*i] = masked_addition(y_shares[(2*i + 5) % new_width], y_shares[2*i]);                                            // z[2*i] = uint4_add(y[(2*i + 5) % new_width], y[2*i]);
        z_shares[2*i + 1] = masked_addition(y_shares[(2*i + 4) % new_width], y_shares[2*i + 1]);                                    // z[2*i + 1] = uint4_add(y[(2*i + 4) % new_width], y[2*i + 1]);
    }

    for (int i = 0; i < new_width; i++) {
        z_shares[i] = masked_addition(z_shares[i], x_shares[(i + 2) % new_width]);                                                  // z[i] = uint4_add(z[i], x[(i + 2) % new_width]);
        z_shares[i] = masked_sbox_second_order(z_shares[i], S_BOXES_B4[i + new_width]);                                             // z[i] = S_BOXES_B4[i + new_width][z[i]];
    }

    for (int i = 0; i < new_width / 3; i++) {
        t_shares[3*i] = masked_addition(masked_addition(z_shares[3*i], z_shares[3*i + 1]), z_shares[3*i + 2]);                      // t[3*i] = uint4_add(uint4_add(z[3*i], z[3*i + 1]), z[3*i + 2]);
        t_shares[3*i + 1] = masked_addition(z_shares[3*i + 1], z_shares[(3*i + 3) % new_width]);                                    // t[3*i + 1] = uint4_add(z[3*i + 1], z[(3*i + 3) % new_width]);
        t_shares[3*i + 2] = masked_addition(masked_addition(z_shares[3*i + 2], z_shares[(3*i + 3) % new_width]), y_shares[3*i]);    // t[3*i + 2] = uint4_add(uint4_add(z[3*i + 2], z[(3*i + 3) % new_width]), y[3*i]);
    }

    t_shares[0] = masked_addition(t_shares[0], x_shares[5]);                                                                        // t[0] = uint4_add(t[0], x[5]);
    t_shares[1] = masked_addition(t_shares[1], x_shares[4]);                                                                        // t[1] = uint4_add(t[1], x[4]);
    t_shares[2] = masked_addition(t_shares[2], x_shares[3]);                                                                        // t[2] = uint4_add(t[2], x[3]);
    t_shares[3] = masked_addition(t_shares[3], x_shares[1]);                                                                        // t[3] = uint4_add(t[3], x[1]);
    t_shares[4] = masked_addition(t_shares[4], x_shares[0]);                                                                        // t[4] = uint4_add(t[4], x[0]);
    t_shares[5] = masked_addition(t_shares[5], x_shares[2]);                                                                        // t[5] = uint4_add(t[5], x[2]);

    packed res_shares = x_shares[new_width];                                                                                        // res = x[new_width];
    for (int i = 0; i < new_width; i++) {
        res_shares = masked_addition(res_shares, masked_sbox_second_order(t_shares[i], S_BOXES_B4[i + 2*new_width]));               // res = uint4_add(res, S_BOXES_B4[i + 2*new_width][t[i]]);
    }
    return res_shares;
}
