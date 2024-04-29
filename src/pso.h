#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    }
}

struct xyPair
{
    double x;
    double y;
};

struct PositionValue
{
    xyPair pos;
    double val;
};

// result - the output of the particle swarm optimizer
// numXS - number of swarms in X direction
// numYS - number of swarms in Y direction
// numP - number of particles
// iterations - number of iterations to simulate
// velW - the weight of the current velocity
// cogAccel - cognitive acceleration constant
// socAccel - social acceleration constant
// lower - the lower bound of the function
// upper - the upper bound of the function
void multiSwarmOptimizer(PositionValue *result, unsigned int numXS, unsigned int numYS,
                         int numP, int iterations, double velW,
                         double cogAccel, double socAccel, double lower, double upper);
