#include "pso.h"
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <ctime>

#define FN_TO_OPTIMIZE(x, y) holderTable(x, y)

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

__device__ double sixHumpCamelFunction(double x, double y)
{
    return (4.0 - 2.1*x*x + pow(x, 4)/3.0)*x*x + x*y + (-4 + 4*y*y)*y*y;
}

// Uniform random function that uses cuda_uniform_double and returns a value between a range
__device__ double cu_urand(double lowerBound, double upperBound, curandState_t *state)
{
    return lowerBound + (upperBound - lowerBound) * curand_uniform_double(state);
}

__global__ void particle(PositionValue *result, int iterations, double velW, double cogAccel, double socAccel,
                         int numXS, int numYS, double lower, double upper, curandState_t *state, time_t seedVal)
{
    __shared__ PositionValue globalBest;
    __shared__ int curIteration;
    __shared__ xyPair s_low;
    __shared__ xyPair s_upp;

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    curand_init(seedVal, idx, 0, &state[idx]);
    
    if(threadIdx.x == 0)
    {
        xyPair interval = {(upper - lower)/numXS, (upper - lower)/numYS};
        s_upp.x = upper - interval.x * (numXS - 1 - blockIdx.x);
        s_low.x = lower + interval.x * blockIdx.x;
        s_upp.y = upper - interval.y * (numYS - 1 - blockIdx.y);
        s_low.y = lower + interval.y * blockIdx.y;
        curIteration = 0;
        globalBest = {cu_urand(s_low.x, s_upp.x, state + idx),
                      cu_urand(s_low.y, s_upp.y, state + idx),
                      0};
        globalBest.val = FN_TO_OPTIMIZE(globalBest.pos.x, globalBest.pos.y);
    }
    __syncthreads();

    PositionValue curr = {cu_urand(s_low.x, s_upp.x, state + idx),
                          cu_urand(s_low.y, s_upp.y, state + idx),
                          0};
    curr.val = FN_TO_OPTIMIZE(curr.pos.x, curr.pos.y);
    
    PositionValue localBest = curr;

    xyPair velocity = {cu_urand(0, 2, state + idx), cu_urand(0, 2, state + idx)};

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

        velocity.x = velW * velocity.x + cogAccel * cu_urand(0, 1, state + idx) * (localBest.x - curr.x) + socAccel * cu_urand(0, 1, state + idx) * (globalBest.x - curr.x);
        velocity.y = velW * velocity.y + cogAccel * cu_urand(0, 1, state + idx) * (localBest.y - curr.y) + socAccel * cu_urand(0, 1, state + idx) * (globalBest.y - curr.y);

        curr.pos.x += velocity.x;
        curr.pos.y += velocity.y;

        if(curr.pos.x < s_low.x)
            curr.pos.x = s_low.x;
        else if(curr.pos.x > s_upp.x)
            curr.pos.x = s_upp.x;

        if(curr.pos.y < s_low.y)
            curr.pos.y = s_low.y;
        else if(curr.pos.y > s_upp.y)
            curr.pos.y = s_upp.y;

        curr.val = FN_TO_OPTIMIZE(curr.pos.x, curr.pos.y);

        if(threadIdx.x == 0)
        {
            curIteration++;
        }
    }

    __syncthreads();
    if(threadIdx.x == 0)
    {
        result[0] = globalBest;
    }
}

void multiSwarmOptimizer(PositionValue *result, unsigned int numXS, unsigned int numYS,
                         int numP, int iterations, double velW, double cogAccel, double socAccel,
                         double lower, double upper)
{
    PositionValue *device_results;
    curandState_t *device_state;
    gpuErrchk( cudaMalloc(&device_state, sizeof(curandState_t)) );
    gpuErrchk( cudaMalloc(&device_results, sizeof(PositionValue) * numXS * numYS) );

    particle<<<{numXS, numYS}, numP>>>(device_results, iterations, velW, cogAccel, socAccel, numXS, numYS, lower, upper, device_state, time(0));

    gpuErrchk( cudaMemcpy(result, device_results, sizeof(PositionValue) * numXS * numYS, cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(device_results) );
    gpuErrchk( cudaFree(device_state) );
}
