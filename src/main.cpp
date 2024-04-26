#include <math.h>
#include "gputk.h"

#include "pso.h"

#define NUM_SWARMS 1
#define NUM_PARTICLES 1024
#define ITERATIONS 50
#define VEL_WEIGHT 1
#define COG_ACCEL  2
#define SOC_ACCEL  2
#define LOWER -10
#define UPPER  10

int main(int argc, char **argv)
{
    PositionValue *result = (PositionValue *)calloc(NUM_SWARMS, sizeof(PositionValue));

    multiSwarmOptimizer(result, NUM_SWARMS, NUM_PARTICLES, ITERATIONS,
                        VEL_WEIGHT, COG_ACCEL, SOC_ACCEL, LOWER, UPPER);

    printf("The answer after %d iterations and %d particles is at (%f, %f) with %f as the min val\n",
            ITERATIONS, NUM_PARTICLES, result[0].x, result[0].y, result[0].val);

    free(result); 

    return 0;
}
