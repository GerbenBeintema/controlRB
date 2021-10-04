#ifndef KERNELS_CU
#define KERNELS_CU

#include "kernels.h"


int h_gridLim_bulk = (NX * NY * NSIM); // out-of-range limit for bulk displace
__constant__ int d_gridLim_bulk;

int h_nxny = (NX * NY); // use in idx calculation
__constant__ int d_nxny;

int h_nxp2nyp2 = (NXP2 * NYP2);
__constant__ int d_nxp2nyp2;


data_t h_tau = tau;
data_t h_tau_t = taut;
data_t h_tau_R = tauR;

data_t h_invtau = 1.0/h_tau;
data_t h_invtau_t = 1.0/h_tau_t;

__constant__ data_t d_tau, d_tau_t, d_tau_R;
__constant__ data_t d_invtau, d_invtau_t;

// other constants
data_t h_alpha_G = alphag;
__constant__ data_t d_alpha_G;

data_t h_ffbody = ff_body;
__constant__ data_t d_ffbody;

data_t h_t0 = T0;
__constant__ data_t d_t0;

// lattice constants
data_t h_cs2 = 1./3.;  __constant__ data_t d_cs2;
data_t h_cs22 = 2./3.; __constant__ data_t d_cs22;
data_t h_cssq = 2./9.; __constant__ data_t d_cssq;
data_t h_cs2i = 3.;    __constant__ data_t d_cs2i;

data_t h_rt0 = 4./9.;  __constant__ data_t d_rt0;
data_t h_rt1 = 1./9.;  __constant__ data_t d_rt1;
data_t h_rt2 = 1./36.; __constant__ data_t d_rt2;


void gpu_mem_init(gpuVars *gpu)
{
	#ifndef SILENT
	fprintf(stderr, "Allocating GPU memory ... \n");
	#endif
	gpu[0].cv = (cudaVars *) malloc (sizeof(cudaVars));

	gpu[0].hv = (hostVars *) malloc (sizeof(hostVars));



	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_threadLookupArr, sizeof(int) * NY * NX * NSIM));

	//checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_tmp_pointer,+-));

	// allocate pops
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f0, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f1, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f2, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f3, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f4, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f5, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f6, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f7, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f8, sizeof(data_t) * NYP2 * NXP2 * NSIM));

	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f0m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f1m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f2m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f3m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f4m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f5m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f6m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f7m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f8m, sizeof(data_t) * NYP2 * NXP2 * NSIM));

	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft0, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft1, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft2, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft3, sizeof(data_t) * NYP2 * NXP2 * NSIM));

	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft0m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft1m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft2m, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft3m, sizeof(data_t) * NYP2 * NXP2 * NSIM));

	// reference pop fields for reset
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f0ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f1ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f2ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f3ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f4ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f5ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f6ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f7ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_f8ref, sizeof(data_t) * NYP2 * NXP2));

	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft0ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft1ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft2ref, sizeof(data_t) * NYP2 * NXP2));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->d_ft3ref, sizeof(data_t) * NYP2 * NXP2));



	// macro fields on device, corresponding host fields alloc in main
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_u, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_v, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_rho, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_temperature, sizeof(data_t) * NYP2 * NXP2 * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_bottemp, sizeof(data_t) * (NX + 2) * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_toptemp, sizeof(data_t) * (NX + 2) * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_time, sizeof(unsigned long int) * NSIM));
#ifdef ACCELERATION
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_eps, sizeof(data_t) * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_omega, sizeof(data_t) * NSIM));
	checkCudaErrors(cudaMalloc((void **) &gpu[0].cv->dev_epssin, sizeof(data_t) * NSIM));
#endif

	// copy constants to memory
	checkCudaErrors( cudaMemcpyToSymbol(d_gridLim_bulk, &h_gridLim_bulk,  sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_nxny,     &h_nxny,  sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_nxp2nyp2, &h_nxp2nyp2,  sizeof(int), 0, cudaMemcpyHostToDevice));

	checkCudaErrors( cudaMemcpyToSymbol(d_tau,      &h_tau,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_tau_t,    &h_tau_t,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_tau_R,    &h_tau_R,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_invtau,   &h_invtau,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_invtau_t, &h_invtau_t,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_alpha_G,  &h_alpha_G,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_ffbody,   &h_ffbody,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_t0,       &h_t0,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_cs2,      &h_cs2,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_cs22,     &h_cs22,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_cssq,     &h_cssq,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_cs2i,     &h_cs2i,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_rt0,      &h_rt0,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_rt1,      &h_rt1,  sizeof(data_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpyToSymbol(d_rt2,      &h_rt2,  sizeof(data_t), 0, cudaMemcpyHostToDevice));

}

void gpu_mem_release(gpuVars *gpu)
{
	#ifndef SILENT
	fprintf(stderr, "Freeing GPU memory ... \n");
	#endif

	checkCudaErrors(cudaFree(gpu[0].cv->dev_threadLookupArr));
	//checkCudaErrors(cudaFree(gpu[0].cv->dev_tmp_pointer));

	checkCudaErrors(cudaFree(gpu[0].cv->d_f0));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f1));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f2));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f3));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f4));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f5));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f6));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f7));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f8));

	checkCudaErrors(cudaFree(gpu[0].cv->d_f0m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f1m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f2m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f3m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f4m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f5m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f6m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f7m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f8m));

	checkCudaErrors(cudaFree(gpu[0].cv->d_ft0));
	checkCudaErrors(cudaFree(gpu[0].cv->d_ft1));
	checkCudaErrors(cudaFree(gpu[0].cv->d_ft2));
	checkCudaErrors(cudaFree(gpu[0].cv->d_ft3));

	checkCudaErrors(cudaFree(gpu[0].cv->d_ft0m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_ft1m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_ft2m));
	checkCudaErrors(cudaFree(gpu[0].cv->d_ft3m));

	checkCudaErrors(cudaFree(gpu[0].cv->d_f0ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f1ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f2ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f3ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f4ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f5ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f6ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f7ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_f8ref));

	checkCudaErrors(cudaFree(gpu[0].cv->d_ft0ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_ft1ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_ft2ref));
	checkCudaErrors(cudaFree(gpu[0].cv->d_ft3ref));

	checkCudaErrors(cudaFree(gpu[0].cv->dev_u));
	checkCudaErrors(cudaFree(gpu[0].cv->dev_v));
	checkCudaErrors(cudaFree(gpu[0].cv->dev_rho));
	checkCudaErrors(cudaFree(gpu[0].cv->dev_temperature));
	checkCudaErrors(cudaFree(gpu[0].cv->dev_bottemp));
	checkCudaErrors(cudaFree(gpu[0].cv->dev_toptemp));
	checkCudaErrors(cudaFree(gpu[0].cv->dev_time));
	#ifdef ACCELERATION
		checkCudaErrors(cudaFree(gpu[0].cv->dev_eps));
		checkCudaErrors(cudaFree(gpu[0].cv->dev_omega));
		checkCudaErrors(cudaFree(gpu[0].cv->dev_epssin));
	#endif
}



/////////////////////////////////////////

// __global__ void gpu_hello_world(data_t *vec)
// {
// 	int index = blockIdx.x * blockDim.x + threadIdx.x;

// 	if(index < NY)
// 	{
// 		vec[index] *= 2.0;
// 	}
// }



// lookup table for threadID to idx mapping done once at sim start
// maybe faster than 14 ops
// need to benchmark to be certain

__global__ void preCompute_idxMap_dispCollide(int *thread_arr)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < d_gridLim_bulk)
	{
		int s = tid / (NX * NY);             // tid/(NY*NX); simulation number
		int i = tid/NY - s*NX;           // collom (x) number
		int j = tid - (i*NY) - s*(NX * NY); // row (y) number

		int idx = (j+1) + (i+1)*NYP2 + s*(NXP2 * NYP2); // maybe dangerous, revisit if buggy

		thread_arr[tid] = idx;
	}
}




__global__ void displace_collide(const data_t * __restrict__ d_f0,
								const data_t * __restrict__ d_f1,
								const data_t * __restrict__ d_f2,
								const data_t * __restrict__ d_f3,
								const data_t * __restrict__ d_f4,
								const data_t * __restrict__ d_f5,
								const data_t * __restrict__ d_f6,
								const data_t * __restrict__ d_f7,
								const data_t * __restrict__ d_f8,
								const data_t * __restrict__ d_ft0,
								const data_t * __restrict__ d_ft1,
								const data_t * __restrict__ d_ft2,
								const data_t * __restrict__ d_ft3,
								data_t *d_f0m,
								data_t *d_f1m,
								data_t *d_f2m,
								data_t *d_f3m,
								data_t *d_f4m,
								data_t *d_f5m,
								data_t *d_f6m,
								data_t *d_f7m,
								data_t *d_f8m,
								data_t *d_ft0m,
								data_t *d_ft1m,
								data_t *d_ft2m,
								data_t *d_ft3m,
								data_t *u_arr,data_t *v_arr,data_t *rho_arr,data_t *T_arr,
								const int * __restrict__ thread_arr,
#ifdef ACCELERATION
								const data_t * __restrict__ epssin,
#endif
								int ioflag)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < d_gridLim_bulk)
	{
		// int s = tid / d_nxny;             // tid/(NY*NX); simulation number
		// int i = tid/NY - s*NX;            // collom (x) number
		// int j = tid - (i*NY) - s*d_nxny;  // row (y) number
		// int idx = (j+1) + (i+1)*NYP2 + s*d_nxp2nyp2; // maybe dangerous, revisit if buggy

		int idx = __ldg(&(thread_arr[tid]));
#ifdef ACCELERATION
		int s = tid / (NX*NY);
#endif
		data_t f0, f1, f2, f3, f4, f5, f6, f7, f8;
		data_t ft0, ft1, ft2, ft3;

		f0 = __ldg(&(d_f0[idx]));
		f1 = __ldg(&(d_f1[idx - NYP2]));
		f2 = __ldg(&(d_f2[idx - 1]));
		f3 = __ldg(&(d_f3[idx + NYP2]));
		f4 = __ldg(&(d_f4[idx + 1]));
		f5 = __ldg(&(d_f5[idx - NYP2 - 1]));
		f6 = __ldg(&(d_f6[idx + NYP2 - 1]));
		f7 = __ldg(&(d_f7[idx + NYP2 + 1]));
		f8 = __ldg(&(d_f8[idx - NYP2 + 1]));

		ft0 = __ldg(&(d_ft0[idx - NYP2]));
		ft1 = __ldg(&(d_ft1[idx - 1]));
		ft2 = __ldg(&(d_ft2[idx + NYP2]));
		ft3 = __ldg(&(d_ft3[idx + 1]));

		data_t T, rho, u, v, ff_buoy, ff_bodyrho, invrho;

		T = ft0 + ft1 + ft2 + ft3;
		rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

		invrho = 1.0/rho;

#ifdef ACCELERATION
		ff_buoy = rho * d_alpha_G * (T - d_t0) * (1. + epssin[s]);
#else
		ff_buoy = rho * d_alpha_G * (T - d_t0);
#endif
		ff_bodyrho = d_ffbody * rho;

		u = ((f1 + f5 + f8 - f3 - f6 - f7) + 0.5 * ff_bodyrho) * invrho;

		v = ((f2 + f5 + f6 - f4 - f7 - f8) + 0.5 * ff_buoy) * invrho;

		data_t u2, v2, sumsq, sumsq2, u2d2cs4, v2d2cs4, ui, vi, uv;

		u2 = u * u;
		v2 = v * v;

		sumsq  = (u2 + v2) / d_cs22;
		sumsq2 = sumsq * (1.0 - d_cs2) * d_cs2i;
		u2d2cs4 = u2 / d_cssq; // cssq = 2/9 = 2*cs2**2
		v2d2cs4 = v2 / d_cssq;

		ui = u * d_cs2i;
		vi = v * d_cs2i;
		uv = ui * vi;

		data_t feq0, feq1, feq2, feq3, feq4, feq5, feq6, feq7, feq8;

		feq0 = rho * d_rt0 * (1.0 - sumsq);
		feq1 = rho * d_rt1 * (1.0 - sumsq + u2d2cs4 + ui);
		feq2 = rho * d_rt1 * (1.0 - sumsq + v2d2cs4 + vi);
		feq3 = rho * d_rt1 * (1.0 - sumsq + u2d2cs4 - ui);
		feq4 = rho * d_rt1 * (1.0 - sumsq + v2d2cs4 - vi);
		feq5 = rho * d_rt2 * (1.0 + sumsq2 + ui + vi + uv);
		feq6 = rho * d_rt2 * (1.0 + sumsq2 - ui + vi - uv);
		feq7 = rho * d_rt2 * (1.0 + sumsq2 - ui - vi + uv);
		feq8 = rho * d_rt2 * (1.0 + sumsq2 + ui - vi - uv);

		// source term
		data_t tauRdrho = d_cs2i * d_tau_R * invrho;
		data_t Rterm = -(ff_bodyrho*u) - (ff_buoy*v);

		d_f0m[idx] = f0 - d_invtau * (f0 - feq0) + (feq0 * tauRdrho * Rterm);
		d_f1m[idx] = f1 - d_invtau * (f1 - feq1) + (feq1 * tauRdrho * (Rterm + ff_bodyrho));
		d_f2m[idx] = f2 - d_invtau * (f2 - feq2) + (feq2 * tauRdrho * (Rterm + ff_buoy));
		d_f3m[idx] = f3 - d_invtau * (f3 - feq3) + (feq3 * tauRdrho * (Rterm - ff_bodyrho));
		d_f4m[idx] = f4 - d_invtau * (f4 - feq4) + (feq4 * tauRdrho * (Rterm - ff_buoy));
		d_f5m[idx] = f5 - d_invtau * (f5 - feq5) + (feq5 * tauRdrho * (Rterm + ff_bodyrho + ff_buoy));
		d_f6m[idx] = f6 - d_invtau * (f6 - feq6) + (feq6 * tauRdrho * (Rterm - ff_bodyrho + ff_buoy));
		d_f7m[idx] = f7 - d_invtau * (f7 - feq7) + (feq7 * tauRdrho * (Rterm - ff_bodyrho - ff_buoy));
		d_f8m[idx] = f8 - d_invtau * (f8 - feq8) + (feq8 * tauRdrho * (Rterm + ff_bodyrho - ff_buoy));


		d_ft0m[idx] = ft0 - d_invtau_t*(ft0 - 0.25*T*(1.+2.*u));
		d_ft1m[idx] = ft1 - d_invtau_t*(ft1 - 0.25*T*(1.+2.*v));
		d_ft2m[idx] = ft2 - d_invtau_t*(ft2 - 0.25*T*(1.-2.*u));
		d_ft3m[idx] = ft3 - d_invtau_t*(ft3 - 0.25*T*(1.-2.*v));

		if(ioflag == 1)
		{
			u_arr[idx] = u;
			v_arr[idx] = v;
			rho_arr[idx] = rho;
			T_arr[idx] = T;
		}

	}
}

__global__ void calculate_macro(const data_t * __restrict__ d_f0,
								const data_t * __restrict__ d_f1,
								const data_t * __restrict__ d_f2,
								const data_t * __restrict__ d_f3,
								const data_t * __restrict__ d_f4,
								const data_t * __restrict__ d_f5,
								const data_t * __restrict__ d_f6,
								const data_t * __restrict__ d_f7,
								const data_t * __restrict__ d_f8,
								const data_t * __restrict__ d_ft0,
								const data_t * __restrict__ d_ft1,
								const data_t * __restrict__ d_ft2,
								const data_t * __restrict__ d_ft3,
								data_t *u_arr, data_t *v_arr, data_t *rho_arr, data_t *T_arr,
								const int * __restrict__ thread_arr)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < d_gridLim_bulk)
	{
		// int s = tid / d_nxny;             // tid/(NY*NX); simulation number
		// int i = tid/NY - s*NX;            // collom (x) number
		// int j = tid - (i*NY) - s*d_nxny;  // row (y) number
		// int idx = (j+1) + (i+1)*NYP2 + s*d_nxp2nyp2; // maybe dangerous, revisit if buggy

		int idx = __ldg(&(thread_arr[tid]));

		data_t f0, f1, f2, f3, f4, f5, f6, f7, f8;
		data_t ft0, ft1, ft2, ft3;

		f0 = __ldg(&(d_f0[idx]));
		f1 = __ldg(&(d_f1[idx]));
		f2 = __ldg(&(d_f2[idx]));
		f3 = __ldg(&(d_f3[idx]));
		f4 = __ldg(&(d_f4[idx]));
		f5 = __ldg(&(d_f5[idx]));
		f6 = __ldg(&(d_f6[idx]));
		f7 = __ldg(&(d_f7[idx]));
		f8 = __ldg(&(d_f8[idx]));

		ft0 = __ldg(&(d_ft0[idx]));
		ft1 = __ldg(&(d_ft1[idx]));
		ft2 = __ldg(&(d_ft2[idx]));
		ft3 = __ldg(&(d_ft3[idx]));

		//data_t T, rho, u, v, ff_buoy, ff_bodyrho, invrho;
		data_t T, rho, u, v, invrho;
		T = ft0 + ft1 + ft2 + ft3;
		rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
		//ff_buoy = rho * d_alpha_G * (T - d_t0);
		//ff_bodyrho = d_ffbody * rho;
		invrho = 1.0/rho;
		//u = ((f1 + f5 + f8 - f3 - f6 - f7) + 0.5 * ff_bodyrho) * invrho; // different?
		//v = ((f2 + f5 + f6 - f4 - f7 - f8) + 0.5 * ff_buoy) * invrho;
		u = (f1 + f5 + f8 - f3 - f6 - f7) * invrho; // different but more simple
		v = (f2 + f5 + f6 - f4 - f7 - f8) * invrho;

		u_arr[idx] = u;
		v_arr[idx] = v;
		rho_arr[idx] = rho;
		T_arr[idx] = T;


	}
}





__global__ void bc_top_bottom_walls(data_t *d_f2, data_t *d_f4,
									data_t *d_f5, data_t *d_f6,
									data_t *d_f7, data_t *d_f8,
									data_t *d_ft1, data_t *d_ft3,
									data_t *bottemp, data_t *toptemp)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < (NXP2 * NSIM - 2) )
	{	
		int bot_idx = (index + 1) * NYP2 + 1;  // bottom wall
		int top_idx = (index + 1) * NYP2 + NY; // top wall

		// at bottom
		d_f2[bot_idx - 1]        = d_f4[bot_idx];
		d_f5[bot_idx - NYP2 - 1] = d_f7[bot_idx];
		d_f6[bot_idx + NYP2 - 1] = d_f8[bot_idx];

		// at top
		d_f4[top_idx + 1]        = d_f2[top_idx];
		d_f8[top_idx - NYP2 + 1] = d_f6[top_idx];
		d_f7[top_idx + NYP2 + 1] = d_f5[top_idx];


		// temperature bottom
		d_ft1[bot_idx - 1] = bottemp[index + 1] * 0.25;

		// temperature top
		d_ft3[top_idx + 1] = toptemp[index + 1] * 0.25;

	}
}




__global__ void bc_east_west_walls_periodic(data_t *d_f1, data_t *d_f3,
											data_t *d_f5, data_t *d_f6,
											data_t *d_f7, data_t *d_f8,
											data_t *d_ft0, data_t *d_ft2)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < (NY * NSIM))
	{
		// For east wall (right side)
		int s = index/NY;
		int j = index - (s*NY) + 1;

		int idx_e_wall  = j + (NX * NYP2) + (s * NXP2 * NYP2);
		int idx_e_ghost = idx_e_wall + NYP2;

		// For west wall (left side)
		int idx_w_wall  = idx_e_wall - ((NX - 1) * NYP2);
		int idx_w_ghost = idx_w_wall - NYP2;

		d_f1[idx_w_ghost] = d_f1[idx_e_wall];
		d_f5[idx_w_ghost] = d_f5[idx_e_wall];
		d_f8[idx_w_ghost] = d_f8[idx_e_wall];
		d_ft0[idx_w_ghost] = d_ft0[idx_e_wall];


		d_f3[idx_e_ghost] = d_f3[idx_w_wall];
		d_f6[idx_e_ghost] = d_f6[idx_w_wall];
		d_f7[idx_e_ghost] = d_f7[idx_w_wall];
		d_ft2[idx_e_ghost]= d_ft2[idx_w_wall];
	}

}



__global__ void bc_east_west_walls_walls(data_t *d_f1, data_t *d_f3,
										data_t *d_f5, data_t *d_f6,
										data_t *d_f7, data_t *d_f8,
										data_t *d_ft0, data_t *d_ft2)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < (NY * NSIM))
	{
		// For east wall (right side)
		int s = index/NY;
		int j = index - (s*NY) + 1;

		int idx_e_wall  = j + (NX * NYP2) + (s * NXP2 * NYP2);
		int idx_e_ghost = idx_e_wall + NYP2; // error?

		// For west wall (left side)
		int idx_w_wall  = idx_e_wall - ((NX - 1) * NYP2);
		int idx_w_ghost = idx_w_wall - NYP2;

		//west
		d_f5[idx_w_ghost-1] = d_f7[idx_w_wall];
		d_f1[idx_w_ghost]   = d_f3[idx_w_wall];
		d_f8[idx_w_ghost+1] = d_f6[idx_w_wall];
		d_ft0[idx_w_ghost]  = d_ft2[idx_w_wall];

		//east
		d_f6[idx_e_ghost-1] = d_f8[idx_e_wall];
		d_f3[idx_e_ghost]   = d_f1[idx_e_wall];
		d_f7[idx_e_ghost+1] = d_f5[idx_e_wall];
		d_ft2[idx_e_ghost]  = d_ft0[idx_e_wall];
	}
}

__global__ void inc_time(long unsigned int * dev_time){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < NSIM){
		dev_time[index]++;
	}
}

__global__ void cal_epssin(const data_t * __restrict__ dev_eps,
						   const data_t * __restrict__ dev_omega, 
						   const long unsigned int * __restrict__ dev_time,
						   data_t * dev_epssin) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < NSIM){
		dev_epssin[index] = dev_eps[index]*sin(dev_omega[index]*(((data_t) dev_time[index])+0.5)); // or sin
	}
}


////////////////////////////////////////

// void gpuCall_helloWorld(gpuVars *gpu)
// {
// 	gpu_hello_world<<<gpu[0].blocks_hello, gpu[0].threads_hello>>>(gpu[0].cv->dev_vec);
// }

#ifdef ACCELERATION
void gpuCall_cal_epssin(gpuVars *gpu) 
{
	cal_epssin<<<gpu[0].blocks_persim,gpu[0].threads_persim>>>(gpu[0].cv->dev_eps,
															   gpu[0].cv->dev_omega, 
															   gpu[0].cv->dev_time,
															   gpu[0].cv->dev_epssin);
}
#endif

void gpuCall_inc_time(gpuVars *gpu)
{
	inc_time<<<gpu[0].blocks_persim,gpu[0].threads_persim>>>(gpu[0].cv->dev_time);
}

void gpuCall_thread_idx_mapper(gpuVars *gpu)
{
#ifndef SILENT
	fprintf(stderr, "Initializing tid -> idx mapper \n\n");
#endif
	preCompute_idxMap_dispCollide<<<gpu[0].blocks_dispColl, gpu[0].threads_dispColl>>>(gpu[0].cv->dev_threadLookupArr);
}


void gpuCall_calculate_macro(gpuVars *gpu)
{
#ifndef SILENT
	fprintf(stderr, "in gpuCall_calculate_macro\n");
#endif
	calculate_macro<<<gpu[0].blocks_dispColl, gpu[0].threads_dispColl>>>(gpu[0].cv->d_f0, gpu[0].cv->d_f1, gpu[0].cv->d_f2,
							gpu[0].cv->d_f3, gpu[0].cv->d_f4, gpu[0].cv->d_f5,
							gpu[0].cv->d_f6, gpu[0].cv->d_f7, gpu[0].cv->d_f8,
							gpu[0].cv->d_ft0, gpu[0].cv->d_ft1,
							gpu[0].cv->d_ft2, gpu[0].cv->d_ft3,
							gpu[0].cv->dev_u, gpu[0].cv->dev_v,	gpu[0].cv->dev_rho, gpu[0].cv->dev_temperature,
							gpu[0].cv->dev_threadLookupArr);

}


void gpuCall_displace_collide(gpuVars *gpu, int ioflag)
{

	displace_collide<<<gpu[0].blocks_dispColl, gpu[0].threads_dispColl>>>(gpu[0].cv->d_f0, gpu[0].cv->d_f1, gpu[0].cv->d_f2,
							gpu[0].cv->d_f3, gpu[0].cv->d_f4, gpu[0].cv->d_f5,
							gpu[0].cv->d_f6, gpu[0].cv->d_f7, gpu[0].cv->d_f8,
							gpu[0].cv->d_ft0, gpu[0].cv->d_ft1,
							gpu[0].cv->d_ft2, gpu[0].cv->d_ft3,
							gpu[0].cv->d_f0m, gpu[0].cv->d_f1m, gpu[0].cv->d_f2m,
							gpu[0].cv->d_f3m, gpu[0].cv->d_f4m, gpu[0].cv->d_f5m,
							gpu[0].cv->d_f6m, gpu[0].cv->d_f7m, gpu[0].cv->d_f8m,
							gpu[0].cv->d_ft0m, gpu[0].cv->d_ft1m,
							gpu[0].cv->d_ft2m, gpu[0].cv->d_ft3m,
							gpu[0].cv->dev_u, gpu[0].cv->dev_v,
							gpu[0].cv->dev_rho, gpu[0].cv->dev_temperature,
							gpu[0].cv->dev_threadLookupArr,
#ifdef ACCELERATION
							gpu[0].cv->dev_epssin,
#endif
							ioflag);

}


void gpuCall_bc_top_bottom_walls(gpuVars *gpu)
{
	bc_top_bottom_walls<<<gpu[0].blocks_bcTB, gpu[0].threads_bcTB>>>(gpu[0].cv->d_f2, gpu[0].cv->d_f4,
								gpu[0].cv->d_f5, gpu[0].cv->d_f6,
								gpu[0].cv->d_f7, gpu[0].cv->d_f8,
								gpu[0].cv->d_ft1, gpu[0].cv->d_ft3,
								gpu[0].cv->dev_bottemp, gpu[0].cv->dev_toptemp);
}


void gpuCall_bc_east_west_walls_periodic(gpuVars *gpu)
{
	bc_east_west_walls_periodic<<<gpu[0].blocks_bcEW, gpu[0].threads_bcEW>>>(gpu[0].cv->d_f1, gpu[0].cv->d_f3,
								gpu[0].cv->d_f5, gpu[0].cv->d_f6,
								gpu[0].cv->d_f7, gpu[0].cv->d_f8,
								gpu[0].cv->d_ft0, gpu[0].cv->d_ft2);
}


void gpuCall_bc_east_west_walls_walls(gpuVars *gpu)
{
	bc_east_west_walls_walls<<<gpu[0].blocks_bcEW, gpu[0].threads_bcEW>>>(gpu[0].cv->d_f1, gpu[0].cv->d_f3,
								gpu[0].cv->d_f5, gpu[0].cv->d_f6,
								gpu[0].cv->d_f7, gpu[0].cv->d_f8,
								gpu[0].cv->d_ft0, gpu[0].cv->d_ft2);
}





unsigned int calc_blocks(unsigned int work, unsigned int threads)
{
	unsigned int grid_size = 0;        // Number of blocks

	if(work % threads == 0) grid_size = (work/threads);
	else grid_size = (work/threads) + 1; 
  
	return grid_size;
}



void gpu_set_threads_blocks(gpuVars *gpu, int which_gpu)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, which_gpu);

	int nDevices;
	cudaGetDeviceCount(&nDevices);
#ifndef SILENT
	fprintf(stderr, "\n\n\nDetected %d GPUs\n", nDevices);
	fprintf(stderr, "Using GPU: %d\n", which_gpu);

	fprintf(stderr, "	Device Name: %s\n", prop.name);
	fprintf(stderr, "	CUDA Capability Major/Minor version number: %d.%d\n", prop.major, prop.minor);
#endif
	// find out the arch here
	if(prop.major < 5)
	{
		fprintf(stderr, "\nGPU arch. is pre-Maxwell\n\nGet a new GPU :)\n\nExiting ..."); //error always prints
		exit(0);
	}

	if(prop.major == 7 && prop.minor == 0)  // TITAN V or Tesla V100
	{
		#ifndef SILENT
		fprintf(stderr, "	ARCH : volta\n");
		#endif
		gpu[0].threads_dispColl     = 256;
		gpu[0].threads_bcTB         = 128;
		gpu[0].threads_bcEW         = 128;
		gpu[0].threads_persim       = 128;
		
	}

	if(prop.major == 6 && prop.minor == 0)  // Tesla P100
	{
		#ifndef SILENT
		fprintf(stderr, "	ARCH : pascal (Tesla P100)\n");
		#endif
		gpu[0].threads_dispColl     = 256;
		gpu[0].threads_bcTB         = 128;
		gpu[0].threads_bcEW         = 128;
		gpu[0].threads_persim       = 128;
		
	}

	if(prop.major == 6 && prop.minor == 1)  // GeForce Pascal
	{
		#ifndef SILENT
		fprintf(stderr, "	ARCH : pascal (GeForce)\n");
		#endif
		gpu[0].threads_dispColl     = 256;
		gpu[0].threads_bcTB         = 128;
		gpu[0].threads_bcEW         = 128;
		gpu[0].threads_persim       = 128;
		
	}

	if(prop.major == 5 && prop.minor == 2)  // GeForce Maxwell
	{
		#ifndef SILENT
		fprintf(stderr, "	ARCH : maxwell (GeForce)\n");
		#endif
		gpu[0].threads_dispColl     = 256;
		gpu[0].threads_bcTB         = 128;
		gpu[0].threads_bcEW         = 128;
		gpu[0].threads_persim       = 128;
		
	}

	gpu[0].blocks_dispColl  = calc_blocks((NX * NY * NSIM),  gpu[0].threads_dispColl);
	gpu[0].blocks_bcTB      = calc_blocks((NXP2 * NSIM - 2), gpu[0].threads_bcTB);
	gpu[0].blocks_bcEW      = calc_blocks((NY * NSIM),       gpu[0].threads_bcEW);
	gpu[0].blocks_persim    = calc_blocks((NSIM),            gpu[0].threads_persim);

}



#endif