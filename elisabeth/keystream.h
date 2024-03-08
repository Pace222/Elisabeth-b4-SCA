#ifndef KEYSTREAM_H
#define KEYSTREAM_H

/*#include <pthread.h>*/

#include "generator.h"

#include "types.h"
#include "filtering_4.h"
#include "filtering_b4.h"


void random_whitened_subset(uint4_t*, uint4_t*, rng*, uint32_t*);
uint4_t filter(uint4_t*, int);
/*uint4_t filter_par(uint4_t*);*/

#endif