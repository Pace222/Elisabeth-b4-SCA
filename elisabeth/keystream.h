#ifndef KEYSTREAM_H
#define KEYSTREAM_H

#include "generator.h"

#include "types.h"
#include "masking.h"
#include "filtering_4.h"
#include "filtering_b4.h"

/* Unprotected and protected RWS implementations */
void random_whitened_subset(uint4_t*, const uint4_t*, const rng*);
void masked_random_whitened_subset(masked*, const masked*, const rng*);
void shuffled_random_whitened_subset(uint4_t*, const uint4_t*, const rng*);
void masked_shuffled_random_whitened_subset(masked*, const masked*, const rng*);
void random_whitened_subset_delayed(uint4_t*, const uint4_t*, const rng*);

/* Unprotected and protected filtering function implementations */
uint4_t filter(const uint4_t*, int);
masked masked_filter(const masked*, int);
uint4_t shuffled_filter(const uint4_t*, int);
masked masked_shuffled_filter(const masked*, int);
uint4_t filter_delayed(const uint4_t*, int);

#endif