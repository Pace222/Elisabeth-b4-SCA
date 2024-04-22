#ifndef MASKING_H
#define MASKING_H
#include "types.h"

#define N_SHARES 3

#define masked_addition(output_shares, inp1_shares, inp2_shares) \
    do { \
        for (int macro_loop_counter = 0; macro_loop_counter < N_SHARES; macro_loop_counter++) { \
            (output_shares)[macro_loop_counter] = uint4_add((inp1_shares)[macro_loop_counter], (inp2_shares)[macro_loop_counter]); \
        } \
    } while(0)

#define masked_addition_constant(output_shares, inp1_shares, inp2_constant) \
    do { \
        output_shares[0] = uint4_add(inp1_shares[0], inp2_constant); \
        for (int macro_loop_counter = 1; macro_loop_counter < N_SHARES; macro_loop_counter++) { \
            (output_shares)[macro_loop_counter] = inp1_shares[macro_loop_counter]; \
        } \
    } while(0)

void gen_shares(uint4_t*);
void init_shares(uint4_t*, uint4_t);
uint4_t consume_shares(const uint4_t*);
//void masked_addition(uint4_t*, const uint4_t*, const uint4_t*);
//void masked_addition_constant(uint4_t*, const uint4_t*, uint4_t);
void masked_negation(uint4_t*, const uint4_t*);
void masked_sbox_second_order(uint4_t*, const uint4_t*, const uint4_t*);

#endif