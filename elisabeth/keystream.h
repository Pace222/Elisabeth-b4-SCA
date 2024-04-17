#ifndef KEYSTREAM_H
#define KEYSTREAM_H

/*#include <pthread.h>*/

#include "generator.h"

#include "types.h"
#include "filtering_4.h"
#include "filtering_b4.h"

void random_whitened_subset(uint4_t*, const uint4_t*, const rng*);
uint4_t filter(const uint4_t*, int);
uint4_t protected_filter_b4_mask_everything(const uint4_t*, int);
uint4_t protected_filter_b4_mask_only_input(const uint4_t*, int);
/*uint4_t filter_par(uint4_t*);*/

#endif