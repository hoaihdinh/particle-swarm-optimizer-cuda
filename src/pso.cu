#include "pso.h"
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <ctime>

// Macro to quickly change the function to be optimized
#define FN_TO_OPTIMIZE(x, y) holderTable(x, y)

// ===== List of all the Functions Used to Optimize =====
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

__device__ double ackleyFunction(double x, double y)
{
    return -20.0 * exp( -0.2 * sqrt( (1.0/2.0) * (x*x + y*y) ) ) - exp( (1.0/2.0) * (cos(2*M_PI*x) + cos(2*M_PI*y)) ) + 20.0 + exp(1.0);
}
// ===== End of all the Functions Used to Optimize =====

// Uniform random function that uses cuda_uniform_double and returns a value between a range
__device__ double cu_urand(double lowerBound, double upperBound, curandState_t *state)
{
    return lowerBound + (upperBound - lowerBound) * curand_uniform_double(state);
}

// The actual kernel/particle function
__global__ void particle(PositionValue *results, int iterations, double velW,
                         double cogAccel, double socAccel, int numXS, int numYS,
                         double lower, double upper, curandState_t *state, time_t seedVal)
{
    // All the values used by all particles of the swarm
    __shared__ PositionValue globalBest;
    __shared__ int curIteration;
    __shared__ xyPair s_low; // Lower boundry of the swarm
    __shared__ xyPair s_upp; // Upper boundry of the swarm

    int id = blockIdx.y * gridDim.x + blockIdx.x + threadIdx.x;
    curand_init(seedVal, id, 0, &state[id]);
    
    // Initializes the shared memory used by all particles in a sub-swarm
    if(threadIdx.x == 0)
    {
        // Sets up the boundry used by all particles in the block
        xyPair interval = {(upper - lower)/numXS, (upper - lower)/numYS};
        s_upp.x = upper - interval.x * (numXS - 1 - blockIdx.x);
        s_low.x = lower + interval.x * blockIdx.x;
        s_upp.y = upper - interval.y * (numYS - 1 - blockIdx.y);
        s_low.y = lower + interval.y * blockIdx.y;

        curIteration = 0;

        // Randomly select a global best location within the boundries
        globalBest = {cu_urand(s_low.x, s_upp.x, state + id),
                      cu_urand(s_low.y, s_upp.y, state + id),
                      0};
        globalBest.val = FN_TO_OPTIMIZE(globalBest.pos.x, globalBest.pos.y);
    }

    __syncthreads();

    // Each particle spwans in a random location in the function within their boundries
    PositionValue curr = {cu_urand(s_low.x, s_upp.x, state + id),
                          cu_urand(s_low.y, s_upp.y, state + id),
                          0};
    curr.val = FN_TO_OPTIMIZE(curr.pos.x, curr.pos.y);
    
    PositionValue localBest = curr; // The first location is their best location as of right now

    xyPair velocity = {cu_urand(0, 2, state + id), cu_urand(0, 2, state + id)}; // Set up a random velocity value

    // The core iteration loop
    while(curIteration < iterations)
    {
        // Update best values if a new minimum is found
        if(localBest.val > curr.val)
        {
            localBest = curr;
            if(globalBest.val > localBest.val)
            {
                globalBest = localBest;
            }
        }
        __syncthreads();

        // Update the velocity of the particle using the PSO velocity function
        // v_new = w * vel_old + c1 * r1 * (personalBest - curr) + c2 * r2 * (globalBest - curr)
        // where w is weight, c1 is cogAccel, c2 is socialAccel, and r1, r2 are random numbers from 0 to 1
        velocity.x = velW * velocity.x + cogAccel * cu_urand(0, 1, state + id) * (localBest.pos.x - curr.pos.x) + socAccel * cu_urand(0, 1, state + id) * (globalBest.pos.x - curr.pos.x);
        velocity.y = velW * velocity.y + cogAccel * cu_urand(0, 1, state + id) * (localBest.pos.y - curr.pos.y) + socAccel * cu_urand(0, 1, state + id) * (globalBest.pos.y - curr.pos.y);

        // Change the position of the particle based on the updated velocities
        curr.pos.x += velocity.x;
        curr.pos.y += velocity.y;

        // Prevent the particle from going out of their boundry
        if(curr.pos.x < s_low.x)
            curr.pos.x = s_low.x;
        else if(curr.pos.x > s_upp.x)
            curr.pos.x = s_upp.x;

        if(curr.pos.y < s_low.y)
            curr.pos.y = s_low.y;
        else if(curr.pos.y > s_upp.y)
            curr.pos.y = s_upp.y;

        curr.val = FN_TO_OPTIMIZE(curr.pos.x, curr.pos.y);

        // Only one thread needs to change the current iteration
        if(threadIdx.x == 0)
        {
            curIteration++;
        }
    }

    __syncthreads();

    // Write out the best result obtained by the swarm
    if(threadIdx.x == 0)
    {
        results[id] = globalBest;
    }
}

void multiSwarmOptimizer(PositionValue *results, unsigned int numXS, unsigned int numYS,
                         int numP, int iterations, double velW, double cogAccel, double socAccel,
                         double lower, double upper)
{
    PositionValue *device_results;
    curandState_t *device_state; // Used for the random number generator
    
    gpuErrchk( cudaMalloc(&device_state, sizeof(curandState_t)) );
    gpuErrchk( cudaMalloc(&device_results, sizeof(PositionValue) * numXS * numYS) );
    
    // The creation of the multi-swarm universe
    particle<<<{numXS, numYS}, numP>>>(device_results, iterations, velW, cogAccel, socAccel,
                                       numXS, numYS, lower, upper, device_state, time(0));

    gpuErrchk( cudaDeviceSynchronize() ); 
    gpuErrchk( cudaMemcpy(results, device_results, sizeof(PositionValue) * numXS * numYS, cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(device_results) );
    gpuErrchk( cudaFree(device_state) );
}
