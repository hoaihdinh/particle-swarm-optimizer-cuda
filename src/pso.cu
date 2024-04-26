#include "pso.h"
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <ctime>

#define CU_URAND(x) curand_uniform_double(x)

__constant__ double const_lower[1];
__constant__ double const_upper[1];

 __device__ double function(double x, double y)
{
    return pow((x - 3.14), 2) + pow((y - 2.72), 2) + sin(3*x + 1.41) + sin(4*y - 1.73);
}

__device__ double sphereFunction(double x, double y)
{
    return x*x + y*y;
}

__device__ double holderTable(double x, double y)
{
    return -abs( sin(x) * cos(y) * exp( abs(1.0 - (sqrt(x * x + y * y)/M_PI)) ) );
}

// Uniform random function that uses cuda_uniform_double and returns a value between a range
__device__ double cu_urand(double lowerBound, double upperBound, curandState_t *state)
{
    return lowerBound + (upperBound - lowerBound) * curand_uniform_double(state);
}

__global__ void particle(PositionValue *result, int iterations, double velW,
                         double cogAccel, double socAccel, curandState_t *state, time_t seedVal)
{
    __shared__ PositionValue globalBest;
    __shared__ int curIteration;
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    curand_init(seedVal, idx, 0, &state[idx]);
    
    PositionValue curr = {cu_urand(*const_lower, *const_upper, state + idx),
                          cu_urand(*const_lower, *const_upper, state + idx),
                          0};
    curr.val = holderTable(curr.x, curr.y);
    
    PositionValue localBest = curr;

    double velocityX = cu_urand(0, *const_upper, state + idx);
    double velocityY = cu_urand(0, *const_upper, state + idx);

    curIteration = 0;
    globalBest.val = curr.val;

    while(curIteration < iterations)
    {
        if(localBest.val > curr.val)
        {
            localBest = curr;
            if(globalBest.val > localBest.val)
            {
                globalBest = localBest;
            }
        }
        __syncthreads();

        velocityX = velW * velocityX + cogAccel * cu_urand(0, 1, state + idx) * (localBest.x - curr.x) + socAccel * cu_urand(0, 1, state + idx) * (globalBest.x - curr.x);
        velocityY = velW * velocityY + cogAccel * cu_urand(0, 1, state + idx) * (localBest.y - curr.y) + socAccel * cu_urand(0, 1, state + idx) * (globalBest.y - curr.y);

        curr.x += velocityX;
        curr.y += velocityY;

        if(curr.x < *const_lower)
            curr.x = *const_lower;
        else if(curr.x > *const_upper)
            curr.x = *const_upper;

        if(curr.y < *const_lower)
            curr.y = *const_lower;
        else if(curr.y > *const_upper)
            curr.y = *const_upper;

        curr.val = holderTable(curr.x, curr.y);

        curIteration++;
    }

    __syncthreads();
    if(threadIdx.x == 0)
    {
        result[0] = globalBest;
    }
}

void multiSwarmOptimizer(PositionValue *result, int numS, int numP, int iterations,
                         double velW, double cogAccel, double socAccel,
                         double lower, double upper)
{
    PositionValue *device_results;
    curandState_t *device_state;
    gpuErrchk( cudaMalloc(&device_state, sizeof(curandState_t)) );
    gpuErrchk( cudaMalloc(&device_results, sizeof(PositionValue) * numS) );
    gpuErrchk( cudaMemcpyToSymbol(const_lower, &lower, sizeof(double), 0, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpyToSymbol(const_upper, &upper, sizeof(double), 0, cudaMemcpyHostToDevice) );

    particle<<<numS, numP>>>(device_results, iterations, velW, cogAccel, socAccel, device_state, time(0));

    gpuErrchk( cudaMemcpy(result, device_results, sizeof(PositionValue) * numS, cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(device_results) );
    gpuErrchk( cudaFree(device_state) );
}
