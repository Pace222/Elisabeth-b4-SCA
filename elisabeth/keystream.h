#ifndef KEYSTREAM_H
#define KEYSTREAM_H

/*#include <pthread.h>*/

#include "generator.h"

#include "types.h"
#include "masking.h"
#include "filtering_4.h"
#include "filtering_b4.h"

void random_whitened_subset(uint4_t*, const uint4_t*, const rng*);
void masked_random_whitened_subset(add_packed*, const add_packed*, const rng*);
void shuffled_random_whitened_subset(uint4_t*, const uint4_t*, const rng*);
void masked_shuffled_random_whitened_subset(add_packed*, const add_packed*, const rng*);

uint4_t filter(const uint4_t*, int);
add_packed masked_filter(const add_packed*, int);
uint4_t shuffled_filter(const uint4_t*, int);
add_packed masked_shuffled_filter(const add_packed*, int);
/*uint4_t filter_par(uint4_t*);*/

#endif