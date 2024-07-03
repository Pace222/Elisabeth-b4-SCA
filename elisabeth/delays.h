#ifndef DELAYS_H
#define DELAYS_H
#include <stdlib.h>
#include "types.h"

#define N_OPERATIONS KEYROUND_WIDTH_B4 * 3

extern uint16_t delays[N_OPERATIONS];
extern size_t delay_head;

static inline void delay_operation(void) __attribute__((always_inline, unused));
static inline void delay_operation(){
    /*
     * Based on Paul Stoffregen's implementation
     * for Teensy 3.0 (http://www.pjrc.com/)
     */

    uint16_t n = delays[delay_head++];
    if (n == 0) return;
    asm volatile(
        "L_%=_delay_operation:"       "\n\t"
        "subs   %0, #1"               "\n\t"
        "bne    L_%=_delay_operation" "\n"
        : "+r" (n) :
        : "cc"
    );
    // LOCAL
    /*asm volatile(
        "L_%=_delay_operation:"       "\n\t"
        "sub   $1, %0"                "\n\t"
        "jne    L_%=_delay_operation" "\n"
        : "+r" (n) :
        : "cc"
    );*/
}

void init_chain(uint8_t*, size_t);
void new_encryption(uint4_t*);

#endif