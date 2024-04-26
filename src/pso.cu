#include "pso.h"
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <ctime>

#define X 0
#define Y 1
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
    __shared__ double s_low[2];
    __shared__ double s_upp[2];

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    curand_init(seedVal, idx, 0, &state[idx]);
    
    if(threadIdx.x == 0)
    {
        double interval[2] = {(upper - lower)/numXS, (upper - lower)/numYS};
        s_upp[X] = upper - interval[X]*(numXS - 1 - blockIdx.x);
        s_low[X] = lower + interval[X]*blockIdx.x;
        s_upp[Y] = upper - interval[Y]*(numYS - 1 - blockIdx.y);
        s_low[Y] = lower + interval[Y]*blockIdx.y;
        curIteration = 0;
        globalBest = {cu_urand(s_low[X], s_upp[X], state + idx),
                      cu_urand(s_low[Y], s_upp[Y], state + idx),
                      0};
        globalBest.val = FN_TO_OPTIMIZE(globalBest.x, globalBest.y);
    }
    __syncthreads();

    PositionValue curr = {cu_urand(s_low[X], s_upp[X], state + idx),
                          cu_urand(s_low[Y], s_upp[Y], state + idx),
                          0};
    curr.val = FN_TO_OPTIMIZE(curr.x, curr.y);
    
    PositionValue localBest = curr;

    double velocityX = cu_urand(0, 2, state + idx);
    double velocityY = cu_urand(0, 2, state + idx);

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

        if(curr.x < s_low[X])
            curr.x = s_low[X];
        else if(curr.x > s_upp[X])
            curr.x = s_upp[X];

        if(curr.y < s_low[Y])
            curr.y = s_low[Y];
        else if(curr.y > s_upp[Y])
            curr.y = s_upp[Y];

        curr.val = FN_TO_OPTIMIZE(curr.x, curr.y);

        curIteration++;
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
