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
        {0x00051968, 0x0002250C, 0x000328A8, 0x00061D04},
        {0x000484A5, 0x00003046, 0x0003BD6B, 0x000811CA},
        {0x0006B9C2, 0x0001A465, 0x0001884E, 0x00069DAB},
        {0x0001250B, 0x0006A027, 0x00071D05, 0x0001A1E9},
        {0x00058DE9, 0x00000180, 0x0002B427, 0x00084090},
        {0x0007B02F, 0x000704CC, 0x000091E1, 0x00013D44},
        {0x000339A0, 0x0003B983, 0x00050870, 0x0004888D},
        {0x00060081, 0x0007AC80, 0x0002418F, 0x00009590},
        {0x000581EA, 0x0004AD22, 0x0002C026, 0x000394EE},
        {0x00068D6B, 0x0004250C, 0x0001B4A5, 0x00041D04},
        {0x00050904, 0x0007ACC4, 0x0003390C, 0x0000954C},
        {0x0006A1C8, 0x0001146B, 0x0001A048, 0x00072DA5},
        {0x0006BC45, 0x0002BD2B, 0x000185CB, 0x000584E5},
        {0x0006814A, 0x00031C6E, 0x0001C0C6, 0x000525A2},
        {0x000010E0, 0x00049180, 0x00083130, 0x0003B090},
        {0x00022CC3, 0x00079982, 0x0006154D, 0x0000A88E},
        {0x0001B028, 0x00043DAF, 0x000691E8, 0x00040461},
        {0x00058C4C, 0x0001A082, 0x0002B5C4, 0x0006A18E}
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
    for (int i = 0; i < BLOCK_WIDTH_B4; i++) {
        x[i] = block[i];
    }
    uint4_t y[new_width];
    uint4_t z[new_width];
    uint4_t t[new_width];
    uint4_t res;

    int start_index, final_index, loop_bound, cmp;

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
    for (int i = 0; i < BLOCK_WIDTH_B4; i++) {                                                                                      // for (int i = 0; i < BLOCK_WIDTH_B4; i++) {
        x_shares[i] = block_shares[i];                                                                                              //     x[i] = block[i];
    }                                                                                                                               // }

    packed y_shares[new_width];
    packed z_shares[new_width];
    packed t_shares[new_width];
    packed res_shares;

    int start_index, final_index, loop_bound, cmp;

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
