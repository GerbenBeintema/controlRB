// see brother code of d2q9-cpu for better documentation on the setup/method/algorithm. 
// Here only the cuda implementation will be discussed
//
// This code has been written for the large part by Pinaki Kulmar (written: 2018)
// Where Gerben Beintema used/changed it to fit his needs for the thesis
// 
// There is a important difference to the gpu code:
// the population are split into multiple varibles thus copying them back to a single will result in a different order of dimentions
// for example the velocity populations are
// p(s,x,y,9) for cpu code
// p(9,s,x,y) for cuda code
// This is corrected for in the python code
// this makes the cuda code preform better but also makes it less readable
// 
// important functions of programs:
// * initfull return a simulation struct
// * simsteps(simulation *sim, int n2steps) 
//    - is the main function that steps all independed simulations
//    - It copy the bottemp and toptemp first to device than steps 
//    -  and copies some of the results back (rho,u,v,temp,pt1,pt3)
// * calxyUvrhot(simulation *sim)
//    - copies the entire state back from the device to the host (and calculates rho,u,v,temp again to let them be up to date)
// * copystate(simulation *sim)
//    - copies the populations from the host to device, can for example be used to load from checkpoint

#ifndef MAIN_C
#define MAIN_C


#include "kernels.h"



simulation* initfull()
{
	#ifndef SILENT
	fprintf(stderr, "Entering C shared lib ... \n");
	#endif
	// assign which GPU to use
	int which_gpu = 0;

	which_gpu = DEVICE_ID; // from make file

	// assign GPU
	checkCudaErrors( cudaSetDevice(which_gpu) );

	// initialize GPU container
	gpuVars *gpu = (gpuVars *) malloc (sizeof(gpuVars));

	// deviceProps and set threads and blocks
	gpu_set_threads_blocks(gpu, which_gpu);

	// initalize GPU arrays
	gpu_mem_init(gpu);

	// initialize CPU arrays
#ifndef SILENT
	printf("NSIM=%d\n",NSIM);
	printf("NX=%d\n", NX);
	printf("NY=%d\n", NY);

	printf("NXP2=%d\n", NYP2);
	printf("NYP2=%d\n", NYP2);

	printf("tau=%f\n", tau);
	printf("taut=%f\n", taut);
	printf("tauR=%f\n", tauR);

	printf("alphag=%f\n", alphag);
	printf("ff_body=%f\n",ff_body);
#endif
	
	// macro variables CPU arrays
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->host_u, (sizeof(data_t) * NYP2 * NXP2 * NSIM)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->host_v, (sizeof(data_t) * NYP2 * NXP2 * NSIM)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->host_rho, (sizeof(data_t) * NYP2 * NXP2 * NSIM)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->host_temperature, (sizeof(data_t) * NYP2 * NXP2 * NSIM)));

	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->host_bottemp, (sizeof(data_t) * NXP2 * NSIM)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->host_toptemp, (sizeof(data_t) * NXP2 * NSIM)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->host_time, (sizeof(data_t) * NSIM)));
#ifdef ACCELERATION
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->host_eps, (sizeof(data_t) * NSIM)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->host_omega, (sizeof(data_t) * NSIM)));
#endif
	// ref pop array for reset and init, host
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_f0ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_f1ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_f2ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_f3ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_f4ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_f5ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_f6ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_f7ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_f8ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_ft0ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_ft1ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_ft2ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_ft3ref, (sizeof(data_t) * NYP2*NXP2)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_p, (sizeof(data_t) * NPOP*NYP2*NXP2*NSIM)));
	checkCudaErrors( cudaMallocHost((void**) &gpu[0].hv->h_pt, (sizeof(data_t) * NPOPT*NYP2*NXP2*NSIM)));



	simulation *sim = (simulation *) malloc (sizeof(simulation));

	sim[0].p1 = gpu[0].hv->h_p; // the same as normal but different order of NPOP, NSIM, NXP2, NYP2 need to be transformed to NSIM, NXP2, NYP2, NPOP
	sim[0].pt1 = gpu[0].hv->h_pt; // the same as normal but different order of NPOPT, NSIM, NXP2, NYP2 need to be transformed to  NSIM, NXP2, NYP2, NPOPT
	sim[0].u = gpu[0].hv->host_u;
	sim[0].v = gpu[0].hv->host_v;
	sim[0].rho = gpu[0].hv->host_rho;
	sim[0].temperature = gpu[0].hv->host_temperature;
	sim[0].bottemp = gpu[0].hv->host_bottemp;
	sim[0].toptemp = gpu[0].hv->host_toptemp;
	sim[0].t = gpu[0].hv->host_time;
	sim[0].gpu = gpu;
#ifdef ACCELERATION
	sim[0].eps = gpu[0].hv->host_eps;
	sim[0].omega = gpu[0].hv->host_omega;
#endif


	srand48((long) seednow);
	data_t T;
	int pp,y,x,idx,s;
	for (s=0; s<NSIM; s++) {
	  for (y=0; y<NYP2; y++) {
		for (x=0; x<NXP2; x++) {
		idx = IDX(y, x)+s*NXP2*NYP2;
		for (pp=0; pp<NPOP; pp++) {
			gpu[0].hv->h_p[NSIM*NXP2*NYP2*pp+idx] = rho0/9.0;
		}

		T = (T0+0.01*2.*(0.5-drand48()));
		gpu[0].hv->h_pt[NSIM*NXP2*NYP2*0 + idx] = 0.25*T; // simplification with vx(p1[idx]) = 0
		gpu[0].hv->h_pt[NSIM*NXP2*NYP2*1 + idx] = 0.25*T;
		gpu[0].hv->h_pt[NSIM*NXP2*NYP2*2 + idx] = 0.25*T;
		gpu[0].hv->h_pt[NSIM*NXP2*NYP2*3 + idx] = 0.25*T;

		//T = (T0+0.01*sin(2*M_PI*(x-1.)/(data_t)(NX-1.))); // inital temptature profile
		//T = (T0+0.01*2.*(0.5-drand48())); // inital temptature profile
		//pt1[idx].p[0] = 0.25*T*(1.+2.*vx(p1[idx]));
		//pt1[idx].p[1] = 0.25*T*(1.+2.*vy(p1[idx]));
		//pt1[idx].p[2] = 0.25*T*(1.-2.*vx(p1[idx]));
		//pt1[idx].p[3] = 0.25*T*(1.-2.*vy(p1[idx]));
		}
	  }
	}

	//make this a function
	copystate(sim);


	// make bc and copy
	for (int id=0;id<NXP2*NSIM; id++) {
		gpu[0].hv->host_bottemp[id] = TBOT0;
		gpu[0].hv->host_toptemp[id] = TTOP0;
	}

	for (int s=0;s<NSIM;s++) {
		gpu[0].hv->host_time[s] = 0;
	}

	checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_bottemp, gpu[0].hv->host_bottemp, sizeof(data_t) * NXP2 * NSIM, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_toptemp, gpu[0].hv->host_toptemp, sizeof(data_t) * NXP2 * NSIM, cudaMemcpyHostToDevice));
#ifndef usefloat // I have no clue why this fails when using floats
	checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_time, gpu[0].hv->host_time, sizeof(unsigned long int) * NSIM, cudaMemcpyHostToDevice));
#endif
	
#ifdef ACCELERATION
   for (int si=0; si<NSIM;si++) {
      sim->eps[si] = 0.;
      sim->omega[si] = 0.;
   }
   checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_eps, gpu[0].hv->host_eps, sizeof(data_t) * NSIM, cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_omega, gpu[0].hv->host_omega, sizeof(data_t) * NSIM, cudaMemcpyHostToDevice));
#endif

	// init tid -> idx mapper look up table
	gpuCall_thread_idx_mapper(gpu);

#ifndef SILENT
	fprintf(stderr, "Initialized CUDA shared lib ... \n");

	fprintf(stderr, "calculating macro...\n");
#endif
	calxyUvrhot(sim);
#ifndef SILENT
	fprintf(stderr, "calculated macro\n");
#endif

	return sim;

}

void copystate(simulation *sim) {
	// copystate form host to device
	gpuVars * gpu = sim[0].gpu;
#ifndef usefloat // I have no clue why this fails when using floats
	checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_time, gpu[0].hv->host_time, sizeof(unsigned long int) * NSIM, cudaMemcpyHostToDevice));
#endif

	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_f0, gpu[0].hv->h_p + 0*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_f1, gpu[0].hv->h_p + 1*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_f2, gpu[0].hv->h_p + 2*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_f3, gpu[0].hv->h_p + 3*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_f4, gpu[0].hv->h_p + 4*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_f5, gpu[0].hv->h_p + 5*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_f6, gpu[0].hv->h_p + 6*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_f7, gpu[0].hv->h_p + 7*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_f8, gpu[0].hv->h_p + 8*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_ft0, gpu[0].hv->h_pt + 0*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_ft1, gpu[0].hv->h_pt + 1*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_ft2, gpu[0].hv->h_pt + 2*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->d_ft3, gpu[0].hv->h_pt + 3*NSIM*NXP2*NYP2, sizeof(data_t) * NSIM*NXP2*NYP2, cudaMemcpyHostToDevice));
}


void calxyUvrhot(simulation *sim) {
	// calculates macro quantaties and copies them back to host, also copies populations back to host
	gpuVars * gpu = sim[0].gpu;
	gpuCall_calculate_macro(gpu);
	checkCudaErrors(cudaMemcpy(gpu[0].hv->host_rho, gpu[0].cv->dev_rho, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->host_temperature, gpu[0].cv->dev_temperature, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->host_u, gpu[0].cv->dev_u, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->host_v, gpu[0].cv->dev_v, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_p + 0*NXP2 * NYP2 * NSIM, gpu[0].cv->d_f0, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_p + 1*NXP2 * NYP2 * NSIM, gpu[0].cv->d_f1, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_p + 2*NXP2 * NYP2 * NSIM, gpu[0].cv->d_f2, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_p + 3*NXP2 * NYP2 * NSIM, gpu[0].cv->d_f3, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_p + 4*NXP2 * NYP2 * NSIM, gpu[0].cv->d_f4, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_p + 5*NXP2 * NYP2 * NSIM, gpu[0].cv->d_f5, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_p + 6*NXP2 * NYP2 * NSIM, gpu[0].cv->d_f6, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_p + 7*NXP2 * NYP2 * NSIM, gpu[0].cv->d_f7, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_p + 8*NXP2 * NYP2 * NSIM, gpu[0].cv->d_f8, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_pt + 0*NXP2 * NYP2 * NSIM, gpu[0].cv->d_ft0, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_pt + 1*NXP2 * NYP2 * NSIM, gpu[0].cv->d_ft1, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_pt + 2*NXP2 * NYP2 * NSIM, gpu[0].cv->d_ft2, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_pt + 3*NXP2 * NYP2 * NSIM, gpu[0].cv->d_ft3, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));

}

void clean(simulation *sim)
{
	gpuVars * gpu = sim[0].gpu;
	// free device arrays
	gpu_mem_release(gpu);

	checkCudaErrors( cudaFreeHost(gpu[0].hv->host_u));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->host_v));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->host_rho));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->host_temperature));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->host_bottemp));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->host_toptemp));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->host_time));
#ifdef ACCELERATION
	checkCudaErrors( cudaFreeHost(gpu[0].hv->host_eps));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->host_omega));
#endif
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_f0ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_f1ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_f2ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_f3ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_f4ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_f5ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_f6ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_f7ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_f8ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_ft0ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_ft1ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_ft2ref));
	checkCudaErrors( cudaFreeHost(gpu[0].hv->h_ft3ref));


	checkCudaErrors( cudaDeviceReset());

	free(gpu);
}



void pointerSwap(gpuVars *gpu)
{
	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_f0;
	gpu[0].cv->d_f0            = gpu[0].cv->d_f0m;
	gpu[0].cv->d_f0m           = gpu[0].cv->dev_tmp_pointer;
	
	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_f1;
	gpu[0].cv->d_f1            = gpu[0].cv->d_f1m;
	gpu[0].cv->d_f1m           = gpu[0].cv->dev_tmp_pointer;
	
	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_f2;
	gpu[0].cv->d_f2            = gpu[0].cv->d_f2m;
	gpu[0].cv->d_f2m           = gpu[0].cv->dev_tmp_pointer;
	
	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_f3;
	gpu[0].cv->d_f3            = gpu[0].cv->d_f3m;
	gpu[0].cv->d_f3m           = gpu[0].cv->dev_tmp_pointer;
	
	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_f4;
	gpu[0].cv->d_f4            = gpu[0].cv->d_f4m;
	gpu[0].cv->d_f4m           = gpu[0].cv->dev_tmp_pointer;
	
	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_f5;
	gpu[0].cv->d_f5            = gpu[0].cv->d_f5m;
	gpu[0].cv->d_f5m           = gpu[0].cv->dev_tmp_pointer;
	
	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_f6;
	gpu[0].cv->d_f6            = gpu[0].cv->d_f6m;
	gpu[0].cv->d_f6m           = gpu[0].cv->dev_tmp_pointer;
	
	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_f7;
	gpu[0].cv->d_f7            = gpu[0].cv->d_f7m;
	gpu[0].cv->d_f7m           = gpu[0].cv->dev_tmp_pointer;
	
	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_f8;
	gpu[0].cv->d_f8            = gpu[0].cv->d_f8m;
	gpu[0].cv->d_f8m           = gpu[0].cv->dev_tmp_pointer;

	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_ft0;
	gpu[0].cv->d_ft0           = gpu[0].cv->d_ft0m;
	gpu[0].cv->d_ft0m          = gpu[0].cv->dev_tmp_pointer;

	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_ft1;
	gpu[0].cv->d_ft1           = gpu[0].cv->d_ft1m;
	gpu[0].cv->d_ft1m          = gpu[0].cv->dev_tmp_pointer;

	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_ft2;
	gpu[0].cv->d_ft2           = gpu[0].cv->d_ft2m;
	gpu[0].cv->d_ft2m          = gpu[0].cv->dev_tmp_pointer;

	gpu[0].cv->dev_tmp_pointer = gpu[0].cv->d_ft3;
	gpu[0].cv->d_ft3           = gpu[0].cv->d_ft3m;
	gpu[0].cv->d_ft3m          = gpu[0].cv->dev_tmp_pointer;
	
}

void cpu_inc_time(gpuVars * gpu) {
	for (int s=0;s<NSIM;++s) {
		gpu[0].hv->host_time[s]++;
	}
}



void simsteps(simulation *sim, int n2steps)
{
	gpuVars * gpu = sim[0].gpu;
#ifndef SILENT
	clock_t timerStart, timerNow;
	data_t  elapsed_time;
	timerStart = clock();
#endif
	//checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_time, gpu[0].hv->host_time, sizeof(unsigned long int) * NSIM, cudaMemcpyHostToDevice)); // if host time get changed
	checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_bottemp, gpu[0].hv->host_bottemp, sizeof(data_t) * NXP2 * NSIM, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_toptemp, gpu[0].hv->host_toptemp, sizeof(data_t) * NXP2 * NSIM, cudaMemcpyHostToDevice));
	#ifdef ACCELERATION
		checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_eps, gpu[0].hv->host_eps, sizeof(data_t) * NSIM, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(gpu[0].cv->dev_omega, gpu[0].hv->host_omega, sizeof(data_t) * NSIM, cudaMemcpyHostToDevice));
	#endif
	for (int i = 0; i < 2*n2steps-1; ++i)
	{
		gpuCall_bc_top_bottom_walls(gpu);
		#ifdef PERIODIC
			gpuCall_bc_east_west_walls_periodic(gpu);
		#else
			gpuCall_bc_east_west_walls_walls(gpu);
		#endif
		#ifdef ACCELERATION
			gpuCall_cal_epssin(gpu);
		#endif
		gpuCall_displace_collide(gpu, 0);
		pointerSwap(gpu);
		gpuCall_inc_time(gpu);
		cpu_inc_time(gpu); //maybe out of loop?
	}

	gpuCall_bc_top_bottom_walls(gpu);
	#ifdef PERIODIC
		gpuCall_bc_east_west_walls_periodic(gpu);
	#else
		gpuCall_bc_east_west_walls_walls(gpu);
	#endif
	#ifdef ACCELERATION
		gpuCall_cal_epssin(gpu);
	#endif
	gpuCall_displace_collide(gpu, 1); //update dev_rho, dev_u, dev_v, dev_temperature
	pointerSwap(gpu);
	gpuCall_inc_time(gpu);
	cpu_inc_time(gpu);



#ifndef SILENT
	timerNow = clock(); 
	elapsed_time = (data_t)(timerNow - timerStart) / (CLOCKS_PER_SEC)*1000; // in ms

	fprintf(stderr, "Completed %d steps, in %3.4f ms, time/step = %3.4f msec\n", n2steps*2, 
			elapsed_time, 
			elapsed_time/((data_t) n2steps*2));
#endif
	

	//copy results back to host
	checkCudaErrors(cudaMemcpy(gpu[0].hv->host_rho, gpu[0].cv->dev_rho, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->host_temperature, gpu[0].cv->dev_temperature, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->host_u, gpu[0].cv->dev_u, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->host_v, gpu[0].cv->dev_v, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));

	//To be able to calculate Nusselts number:
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_pt + 1*NXP2 * NYP2 * NSIM, gpu[0].cv->d_ft1, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu[0].hv->h_pt + 3*NXP2 * NYP2 * NSIM, gpu[0].cv->d_ft3, sizeof(data_t) * NXP2 * NYP2 * NSIM, cudaMemcpyDeviceToHost));
	
}


int ctypes_test_that_cals_x_time_x(int x) { // test function
	printf("x=%d\n", x);
	int y;
	y = x*x;
	printf("x**2=%d\n", y);
	return y;
}

int main(const int argc, const char **argv)
{
	// only for testing purposes, see "make test"
    fprintf(stderr, "Starting init\n");
	simulation * sim = initfull();
	int j = NY; 
	int i = NX/2;
	int s = 0;
	int idx = j + i*NYP2 + s*NXP2*NYP2;
	calxyUvrhot(sim);

	fprintf(stderr, "sim->temperature[%d,%d,%d] = %f\n", s,i,j,sim->temperature[idx]);
	fprintf(stderr, "Done with init\n");
	fprintf(stderr, "stepping 1000 steps\n");
	simsteps(sim,500);
	calxyUvrhot(sim);
	fprintf(stderr, "succefull steped\n");
	fprintf(stderr, "sim->temperature[%d,%d,%d] = %f\n", s,i,j,sim->temperature[idx]);



	fprintf(stderr, "Cleaning...\n");
	clean(sim);
	fprintf(stderr, "Done clean\n");

	return 0;
}


#endif