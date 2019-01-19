/*******************************************************************
Author: Nick Seedorff
Date: 1/17/2019
Program Description: Takes arguments of the form "data alpha beta" 
and evaluates the Pareto density. 
*******************************************************************/

#include <stdio.h>
#include <math.h>

/* Function that calculates pareto density */
float pare_dens(float data, float alpha, float beta)
{
  /* Default is 0, change if parameters are negative or data < alpha */ 
  float dens_result = 0;

  if(alpha <= 0 || beta <= 0){
    dens_result = -1; /* Out of parameter space, print a warning in main */
  } else if(data > alpha){
    dens_result = beta * pow(alpha, beta) / pow(data, (beta + 1));
  } 
  return dens_result; /* will return 0 if data is out of the support */
}

/* Takes standard input and prints results of density function */
int main()
{
  float x; /* Input data to evaluate */
  float a; /* Alpha parameter, must be > 0 */
  float b; /* Beta parameter, must be > 0 */
  float result; /* Result of evaluating pareto density */

  /*Search input for three floats*/
  scanf("%f %f %f", &x, &a, &b);
  result = pare_dens(x, a, b);

  if(result == -1){ /* Print warning if invalid parameters*/
    printf("Warning: Density is NaN. Check parameter values.\n");
  } else {
    printf("%f\n", result);
  }
  return 0;
}
