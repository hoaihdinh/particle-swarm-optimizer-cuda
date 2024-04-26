#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    }
}

struct PositionValue
{
    double x;
    double y;
    double val;
};

// result - the output of the particle swarm optimizer
// numS - number of swarms
// numP - number of particles
// iterations - number of iterations to simulate
// velW - the weight of the current velocity
// cogAccel - cognitive acceleration constant
// socAccel - social acceleration constant
// lower - the lower bound of the function
// upper - the upper bound of the function
void multiSwarmOptimizer(PositionValue *result, int numS, int numP, int iterations, double velW,
                         double cogAccel, double socAccel, double lower, double upper);
