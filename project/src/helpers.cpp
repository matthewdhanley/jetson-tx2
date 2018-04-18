//
// Created by matt on 4/17/18.
//

#include "../include/helpers.h"

extern "C" int next_pow_of_two(int val){
    val--;
    val |= val >> 1;
    val |= val >> 2;
    val |= val >> 4;
    val |= val >> 8;
    val |= val >> 16;
    val++;
    val /= 2;
    return val;
}