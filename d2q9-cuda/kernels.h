#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuComplex.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NX // if NX is not defined
	#warning "Using default configuration"
	#define NY (64)
	#define NX (64)
	#define NSIM (8)

	#define tau 0.5666 // time constant velocity nu = cs2*(tau-0.5)
	#define taut 0.5666 // time constant tempature kappa = cs2*(taut-0.5)


	#define alphag 0.000000
	#define ff_body  0.000000
	// #ifndef alphag
	// #warning warning: setting alphag to zero
	// #endif
	// #ifndef ff_body
	// #warning warning: setting ff_body to zero
	// #endif

	/* WARNING vx is rho.vx */

	#ifndef TTOP0
	#define TTOP0 (1.)
	#endif
	#ifndef TBOT0
	#define TBOT0 (2.)
	#endif
	#ifndef T0
	#define T0    (0.5*(TTOP0+TBOT0))
	#endif

	//#define PERIODIC
	#define WALLED
#endif

#ifdef usefloat
	#define data_t float
	//#warning "Using float"
#else
	#define data_t double
	//#warning "Using double"
#endif

#ifndef WALLED //&?
#ifndef PERIODIC
# error ERROR: NO WALL PROP SET !!!
#endif
#endif

#define NPOP 9
#define NPOPT 4
#ifndef seednow // using the name seed for this overwrote something in a libary (and caused crashes) thus now it is named seednow.
#define seednow  42
#endif
#define tauR (1.-1./(2.*tau))
#define NXP2 (NX+2)
#define NYP2 (NY+2)
#define rho0 1.0
#define bceps 0.00150
// #ifndef WALLED //&?
// #ifndef PERIODIC
// # error ERROR: NO WALL PROP SET !!!
// #endif
// #endif
// #define SILENT



#define vx(a) (a.p[1]+a.p[5]+a.p[8]-a.p[3]-a.p[6]-a.p[7]) // velocity x 
#define vy(a) (a.p[2]+a.p[5]+a.p[6]-a.p[4]-a.p[7]-a.p[8]) // velocity y
#define vt(a) (a.p[0]+a.p[1]+a.p[2]+a.p[3]) // tempature
#define  m(a) (a.p[0]+a.p[1]+a.p[2]+a.p[3]+a.p[4]+a.p[5]+a.p[6]+a.p[7]+a.p[8]) // mass

#define IDX(j,i) ((j) + NYP2*(i))


#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

typedef struct {
   data_t p[9];
} pop; //populations struct

typedef struct {
   data_t p[4];
} popt; //population tempature

typedef struct
{
	data_t *dev_vec;

	int *dev_threadLookupArr;

	// 9 pop velocity and move mirror swap
	// each pop has multiple copies for all sims
	data_t *d_f0, *d_f1, *d_f2, *d_f3, *d_f4, *d_f5, *d_f6, *d_f7, *d_f8;
	data_t *d_f0m, *d_f1m, *d_f2m, *d_f3m, *d_f4m, *d_f5m, *d_f6m, *d_f7m, *d_f8m;

	// 4 pop temperature and move mirror swap
	data_t *d_ft0, *d_ft1, *d_ft2, *d_ft3;
	data_t *d_ft0m, *d_ft1m, *d_ft2m, *d_ft3m;

	// reference pops to reset sim
	data_t *d_f0ref, *d_f1ref, *d_f2ref, *d_f3ref, *d_f4ref, *d_f5ref, *d_f6ref, *d_f7ref, *d_f8ref;
	data_t *d_ft0ref, *d_ft1ref, *d_ft2ref, *d_ft3ref;

	// macro quantity vars
	data_t *dev_u, *dev_v, *dev_rho, *dev_temperature;

	// RL control
	data_t *dev_bottemp;
	data_t *dev_toptemp;

#ifdef ACCELERATION
	data_t *dev_eps;
	data_t *dev_omega;
	data_t *dev_epssin;
#endif

	unsigned long int *dev_time;

	data_t *dev_tmp_pointer;

} cudaVars;


typedef struct 
{
	data_t *host_u, *host_v, *host_rho, *host_temperature;
	data_t *host_bottemp;
	data_t *host_toptemp;

#ifdef ACCELERATION
	data_t * host_eps;
	data_t * host_omega;
#endif

	data_t *h_f0ref, *h_f1ref, *h_f2ref, *h_f3ref, *h_f4ref, *h_f5ref, *h_f6ref, *h_f7ref, *h_f8ref;
	data_t *h_ft0ref, *h_ft1ref, *h_ft2ref, *h_ft3ref;

	pop *h_p1ref; 
	popt *h_pt1ref;


	data_t * h_pt;
	data_t * h_p;
	unsigned long int * host_time; // list of time per simulation


} hostVars;

typedef struct
{
	cudaVars *cv;

	hostVars *hv;

	// thread for kernels
	unsigned int threads_dispColl;
	unsigned int threads_bcTB;
	unsigned int threads_bcEW;
	unsigned int threads_persim;

	// blocks for kernels
	unsigned int blocks_dispColl;
	unsigned int blocks_bcTB;
	unsigned int blocks_bcEW;
	unsigned int blocks_persim;

} gpuVars;

typedef struct {
   data_t * p1;
   data_t * p2;
   data_t * pt1;
   data_t * pt2;
   data_t * u;
   data_t * v;
   data_t * rho;
   data_t * temperature;
   data_t * bottemp;
   data_t * toptemp;
   unsigned long int * t;
#ifdef ACCELERATION
   data_t * eps;
   data_t * omega;
#endif
   gpuVars * gpu;
} simulation; //main holder for python



extern "C"
{
	void gpu_set_threads_blocks(gpuVars *gpu, int which_gpu);

	void gpu_mem_init(gpuVars *gpu);

	void gpu_mem_release(gpuVars *gpu);

	void clean(simulation *sim);

	void simsteps(simulation *sim, int n2steps);

	void copystate(simulation *sim);

	void calxyUvrhot(simulation *sim);

	simulation* initfull();

	int ctypes_test_that_cals_x_time_x(int);

	void gpuCall_thread_idx_mapper(gpuVars *gpu);

	void gpuCall_cal_epssin(gpuVars *gpu);

	void gpuCall_inc_time(gpuVars *gpu);

	void gpuCall_calculate_macro(gpuVars *gpu);

	void gpuCall_displace_collide(gpuVars *gpu, int ioflag);

	void gpuCall_bc_top_bottom_walls(gpuVars *gpu);

	void gpuCall_bc_east_west_walls_periodic(gpuVars *gpu);

	void gpuCall_bc_east_west_walls_walls(gpuVars *gpu);

}


#endif
