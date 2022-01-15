#include <stdlib.h>
#include "LPopt.h"
#ifdef __cplusplus
extern "C" {
#endif

int LPp_project_wrapper(double *y, double r, double *x, double *info, int n, double p);
	
#ifdef __cplusplus
}
#endif

int LPp_project_wrapper(double *y, double r, double *x, double *info, int n, double p) {
	return LPp_project(y, r, x, info, n, p, NULL);
}
