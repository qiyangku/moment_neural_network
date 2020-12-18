// Copyright 2020 Wang Dai, ISTBI, Fudan University China

#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Python CPU function
void dawson1_cpu(float* , float* , unsigned int );
void dawson1_int_cpu(float*, float*, unsigned int, double*, int, const int);
//extern void dawson2_cpu(float*, float*, unsigned int, float*, int, float*, const int);
void dawson2_cpu(float*, float*, unsigned int, double*, int, float*, const int);
// extern void dawson2_int_cpu(float*, float*, unsigned int, float*, int, float*, const int);

void dawson2_int_cpu(float*, float*, unsigned int, double*, int, float*, const int);

