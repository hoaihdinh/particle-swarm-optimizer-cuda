#include <math.h>
#include "gputk.h"

#include "pso.h"

#define NUM_SWARMS_X 1
#define NUM_SWARMS_Y 1
#define NUM_PARTICLES 1024
#define ITERATIONS 100
#define VEL_WEIGHT 0.7
#define COG_ACCEL  2
#define SOC_ACCEL  2
#define LOWER -10
#define UPPER  10
#define NUM_SWARMS (NUM_SWARMS_X*NUM_SWARMS_Y)

int main(int argc, char **argv)
{
    PositionValue *results = (PositionValue *)calloc(NUM_SWARMS, sizeof(PositionValue));

    multiSwarmOptimizer(results, NUM_SWARMS_X, NUM_SWARMS_Y, NUM_PARTICLES, ITERATIONS,
                        VEL_WEIGHT, COG_ACCEL, SOC_ACCEL, LOWER, UPPER);

    printf("%d iterations, %d swarm(s), %d particles per swarm\n", ITERATIONS, NUM_SWARMS, NUM_PARTICLES);
    int offset = 0;
    for(int i = 0; i < NUM_SWARMS; i++)
    {
        printf("Swarm %d found their minimum value of %lf at (%lf, %lf)\n",
                i, results[i].val, results[i].pos.x, results[i].pos.y);
    }

    free(results);
    return 0;
}
