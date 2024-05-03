#include <stddef.h>

#include "filtering_b4.h"
#include "masking.h"

uint32_t S_BOXES_B4_PACKED[18][4];

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
    uint32_t s_boxes_32_t[18][4] = {
        {0x0A060B08, 0x0409080C, 0x060A0508, 0x0C070804},
        {0x09010505, 0x000C0206, 0x070F0B0B, 0x10040E0A},
        {0x0D0E0E02, 0x03090305, 0x0302020E, 0x0D070D0B},
        {0x0209080B, 0x0D080107, 0x0E070805, 0x03080F09},
        {0x0B030F09, 0x00000C00, 0x050D0107, 0x10100410},
        {0x0F0C010F, 0x0E01060C, 0x01040F01, 0x020F0A04},
        {0x060E0D00, 0x070E0C03, 0x0A020310, 0x0902040D},
        {0x0C000401, 0x0F0B0400, 0x04100C0F, 0x01050C10},
        {0x0B000F0A, 0x090B0902, 0x05100106, 0x0705070E},
        {0x0D030B0B, 0x0809080C, 0x030D0505, 0x08070804},
        {0x0A020804, 0x0F0B0604, 0x060E080C, 0x01050A0C},
        {0x0D080E08, 0x0205030B, 0x03080208, 0x0E0B0D05},
        {0x0D0F0205, 0x050F090B, 0x03010E0B, 0x0B010705},
        {0x0D000A0A, 0x0607030E, 0x03100606, 0x0A090D02},
        {0x00040700, 0x09040C00, 0x100C0910, 0x070C0410},
        {0x040B0603, 0x0F060C02, 0x0C050A0D, 0x010A040E},
        {0x030C0108, 0x080F0D0F, 0x0D040F08, 0x08010301},
        {0x0B03020C, 0x03080402, 0x050D0E04, 0x0D080C0E}
    };
    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < 16; j++) {
            S_BOXES_B4[i][j] = uint4_new(s_boxes_8_t[i][j]);
        }
    }
    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < 4; j++) {
            S_BOXES_B4_PACKED[i][j] = s_boxes_32_t[i][j];
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

packed masked_filter_block_b4(const packed* block_shares) {
    // Protection under second order DPA
    size_t new_width = BLOCK_WIDTH_B4 - 1;

    packed x_shares[BLOCK_WIDTH_B4];
    for (int i = 0; i < BLOCK_WIDTH_B4; i++) {                                                                                      // for (int i = 0; i < BLOCK_WIDTH_B4; i++) {
        x_shares[i] = block_shares[i];                                                                                              //     x[i] = block[i];
    }                                                                                                                               // }

    packed y_shares[new_width];
    packed z_shares[new_width];
    packed t_shares[new_width];
    packed res_shares;

    for (int i = 0; i < new_width / 2; i++) {
        x_shares[2*i + 1] = masked_addition(x_shares[2*i + 1], x_shares[2*i]);                                                      // x[2*i + 1] = uint4_add(x[2*i + 1], x[2*i]);
    }

    for (int i = 0; i < new_width; i++) {
        y_shares[i] = masked_sbox_first_order(x_shares[i], S_BOXES_B4_PACKED[i]);                                                         // y[i] = S_BOXES_B4[i][x[i]];
    }

    for (int i = 0; i < new_width / 2; i++) {
        z_shares[2*i] = masked_addition(y_shares[(2*i + 5) % new_width], y_shares[2*i]);                                            // z[2*i] = uint4_add(y[(2*i + 5) % new_width], y[2*i]);
        z_shares[2*i + 1] = masked_addition(y_shares[(2*i + 4) % new_width], y_shares[2*i + 1]);                                    // z[2*i + 1] = uint4_add(y[(2*i + 4) % new_width], y[2*i + 1]);
    }

    for (int i = 0; i < new_width; i++) {
        z_shares[i] = masked_addition(z_shares[i], x_shares[(i + 2) % new_width]);                                                  // z[i] = uint4_add(z[i], x[(i + 2) % new_width]);
        z_shares[i] = masked_sbox_first_order(z_shares[i], S_BOXES_B4_PACKED[i + new_width]);                                             // z[i] = S_BOXES_B4[i + new_width][z[i]];
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

    res_shares = x_shares[new_width];                                                                                        // res = x[new_width];
    for (int i = 0; i < new_width; i++) {
        res_shares = masked_addition(res_shares, masked_sbox_first_order(t_shares[i], S_BOXES_B4_PACKED[i + 2*new_width]));               // res = uint4_add(res, S_BOXES_B4[i + 2*new_width][t[i]]);
    }
    return res_shares;
}

uint4_t shuffled_filter_block_b4(const uint4_t* block) {
    size_t new_width = BLOCK_WIDTH_B4 - 1;
    uint4_t x[BLOCK_WIDTH_B4];
    uint4_t y[new_width];
    uint4_t z[new_width];
    uint4_t t[new_width];
    uint4_t res;

    int start_index, final_index, loop_bound, cmp;

    loop_bound = BLOCK_WIDTH_B4;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        x[final_index] = block[final_index];
    }

    loop_bound = new_width / 2;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        x[2*final_index + 1] = uint4_add(x[2*final_index + 1], x[2*final_index]);
    }

    loop_bound = new_width;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        y[final_index] = S_BOXES_B4[final_index][x[final_index]];
    }

    loop_bound = new_width / 2;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        z[2*final_index] = uint4_add(y[(2*final_index + 5) % new_width], y[2*final_index]);
        z[2*final_index + 1] = uint4_add(y[(2*final_index + 4) % new_width], y[2*final_index + 1]);
    }

    loop_bound = new_width;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        z[final_index] = uint4_add(z[final_index], x[(final_index + 2) % new_width]);
        z[final_index] = S_BOXES_B4[final_index + new_width][z[final_index]];
    }

    loop_bound = new_width / 3;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        t[3*final_index] = uint4_add(uint4_add(z[3*final_index], z[3*final_index + 1]), z[3*final_index + 2]);
        t[3*final_index + 1] = uint4_add(z[3*final_index + 1], z[(3*final_index + 3) % new_width]);
        t[3*final_index + 2] = uint4_add(uint4_add(z[3*final_index + 2], z[(3*final_index + 3) % new_width]), y[3*final_index]);
    }

    uint8_t x_order[6] = { 5, 4, 3, 1, 0, 2};
    loop_bound = new_width;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;
        
        t[final_index] = uint4_add(t[final_index], x[x_order[final_index]]);
    }

    res = x[new_width];
    loop_bound = new_width;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        res = uint4_add(res, S_BOXES_B4[final_index + 2*new_width][t[final_index]]);
    }
    return res;
}

packed masked_shuffled_filter_block_b4(const packed* block_shares) {
    // Protection under second order DPA
    size_t new_width = BLOCK_WIDTH_B4 - 1;
    packed x_shares[BLOCK_WIDTH_B4];
    packed y_shares[new_width];
    packed z_shares[new_width];
    packed t_shares[new_width];
    packed res_shares;

    int start_index, final_index, loop_bound, cmp;

    loop_bound = BLOCK_WIDTH_B4;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;
        
        x_shares[final_index] = block_shares[final_index];
    }

    loop_bound = new_width / 2;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        x_shares[2*final_index + 1] = masked_addition(x_shares[2*final_index + 1], x_shares[2*final_index]);                                                      // x[2*i + 1] = uint4_add(x[2*i + 1], x[2*i]);
    }

    loop_bound = new_width;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        y_shares[final_index] = masked_sbox_first_order(x_shares[final_index], S_BOXES_B4_PACKED[final_index]);                                                         // y[i] = S_BOXES_B4[i][x[i]];
    }

    loop_bound = new_width / 2;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        z_shares[2*final_index] = masked_addition(y_shares[(2*final_index + 5) % new_width], y_shares[2*final_index]);                                            // z[2*i] = uint4_add(y[(2*i + 5) % new_width], y[2*i]);
        z_shares[2*final_index + 1] = masked_addition(y_shares[(2*final_index + 4) % new_width], y_shares[2*final_index + 1]);                                    // z[2*i + 1] = uint4_add(y[(2*i + 4) % new_width], y[2*i + 1]);
    }

    loop_bound = new_width;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        z_shares[final_index] = masked_addition(z_shares[final_index], x_shares[(final_index + 2) % new_width]);                                                  // z[i] = uint4_add(z[i], x[(i + 2) % new_width]);
        z_shares[final_index] = masked_sbox_first_order(z_shares[final_index], S_BOXES_B4_PACKED[final_index + new_width]);                                             // z[i] = S_BOXES_B4[i + new_width][z[i]];
    }

    loop_bound = new_width / 3;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        t_shares[3*final_index] = masked_addition(masked_addition(z_shares[3*final_index], z_shares[3*final_index + 1]), z_shares[3*final_index + 2]);                      // t[3*i] = uint4_add(uint4_add(z[3*i], z[3*i + 1]), z[3*i + 2]);
        t_shares[3*final_index + 1] = masked_addition(z_shares[3*final_index + 1], z_shares[(3*final_index + 3) % new_width]);                                    // t[3*i + 1] = uint4_add(z[3*i + 1], z[(3*i + 3) % new_width]);
        t_shares[3*final_index + 2] = masked_addition(masked_addition(z_shares[3*final_index + 2], z_shares[(3*final_index + 3) % new_width]), y_shares[3*final_index]);    // t[3*i + 2] = uint4_add(uint4_add(z[3*i + 2], z[(3*i + 3) % new_width]), y[3*i]);
    }

    uint8_t x_shares_order[6] = { 5, 4, 3, 1, 0, 2};
    loop_bound = new_width;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        t_shares[final_index] = masked_addition(t_shares[final_index], x_shares[x_shares_order[final_index]]);
    }

    res_shares = x_shares[new_width];                                                                                        // res = x[new_width];
    loop_bound = new_width;
    start_index = get_rand();
    for (int i = 0; i < loop_bound; i++) {
        final_index = start_index + i;
        cmp = final_index >= loop_bound;
        final_index -= cmp * loop_bound;

        res_shares = masked_addition(res_shares, masked_sbox_first_order(t_shares[final_index], S_BOXES_B4_PACKED[final_index + 2*new_width]));               // res = uint4_add(res, S_BOXES_B4[i + 2*new_width][t[i]]);
    }
    return res_shares;
}
