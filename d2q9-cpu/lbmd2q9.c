

// This the code by Gerben Beintema (id: 0907710) for his master thesis of AP 2018-2019, 
// Thesis: "Finding noval control statagies for flow using reinforcement learning and high performance simulations"
// Contact g.i.beintema@student.tue.nl and gerben.beintema@gmail.com for more information
// 
// This code has been used as a template for the creation of its sister code d2q9-cuda 
// that implements the same problem but than in cuda
// 
// The code has been setup to be able to be loaded by python as a shared libary (just as the cuda code)
// This has made the connections to control algorithms easier. 
//
// This is lbm code for a 2D with 9 velocity population and 4 temperature populations 
// with and extra dimention for multiple simulations
// and usable interface to python where the temperature of the boundary can be dynemicly changed.
// For details see papers (or contact me by mail)
// 1. 6.4 Alternative Forcing Schemes (p. 241) method: He et al. [39]
// 2. Numerical study of lattice Boltzmann methods for a convection–diffusion equation coupled with Navier–Stokes equations


// Velocity populations, 9 populations
/*********************************************
 *                                           *
 *        x  y                               *
 * 0    (+0,+0)         6    2    5          *
 * 1    (+1,+0)          \   |   /           *
 * 2    (+0,+1)           \  |  /            *
 * 3    (-1,+0)            \ | /             *
 * 4    ( 0,-1)      3 <---- 0 ----> 1       *
 * 5    (+1,+1)            / | \             *
 * 6    (-1,+1)           /  |  \            *
 * 7    (-1,-1)          /   |   \           *
 * 8    (+1,-1)         7    4    8          * 
 *                                           *
 *********************************************/

// Temperature populations,  4 populations
/*********************************************
 *                                           *
 *        x  y                               *
 * 0    (+1,+0)              1               *
 * 1    (+0,+1)              |               *
 * 2    (-1,+0)              |               *
 * 3    (+0,-1)              |               *
 *                   2 <-----+----> 0        *
 *                           |               *
 *                           |               *
 *                           |               *
 *                           3               * 
 *                                           *
 *********************************************/

// ### SIMULATION SPACE ####
// #
// # dimentions in simulation are NSIM, NXP2 (NX + 2), NYP2 (NY + 2),... (first s, x than y)
// # the ith simuation in the simulation dimension is denoted by a 's' 
// # thus for example u(s,x,y) has the size of (NSIM,NXP2,NYP2) 
// #
// #                                      Top boundary (no slip tempature controlable)
// #                                    0,NY+1.............. NX+1,NY+1
// #y                                   .                  .
// #^                                   .                  .
// #|noslip-adiabatic-wall (or periodic).      domain      . noslip-adiabatic-wall  (or periodic)
// #|                                   .                  .
// #|                                   .                  .
// #|                                   0,0.................0,NX+1
// #|                                     bottom boundary (no slip tempature controlable)
// #+-----> x
// # with s (simulation dimention) being a the first dimintion 


#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "defs.h" // get definitions of the current simulation


typedef struct {
   data_t p[9];
} pop; //populations velocity

typedef struct {
   data_t p[4];
} popt; //population tempature

typedef struct { 
   pop * p1;
   pop * p2;
   popt * pt1;
   popt * pt2;
   data_t * u;
   data_t * v;
   data_t * rho;
   data_t * temperature;
   data_t * bottemp;
   data_t * toptemp;
   unsigned long int * t; // list of time per simulation
#ifdef ACCELERATION
   data_t * eps;
   data_t * omega;
#endif
   void * gpu;
} simulation; //simulation struct that is passed to python to be able to parce and change the data

#include "discolbckernals.h"
#include "utils.h"

void lonesteps(simulation * sim, int s, int n2steps) { // step simulation s for 2*n2steps, assuming p1,p1t is up to date
   pop * p1, * p2;
   popt * pt1, * pt2;
   data_t * bottemp, * toptemp;
   unsigned long int * t;

   p1 = sim->p1+s*NXP2*NYP2; // Load simulation s by offsetting the pointer
   p2 = sim->p2+s*NXP2*NYP2;
   pt1 = sim->pt1+s*NXP2*NYP2;
   pt2 = sim->pt2+s*NXP2*NYP2;
   bottemp = sim->bottemp + s*NXP2;
   toptemp = sim->toptemp + s*NXP2;
   t = sim->t+s;

#ifdef ACCELERATION
   data_t eps,omega,epssin;
   eps = sim->eps[s];
   omega = sim->omega[s];
#endif
   pop * ptemp; //pointer used for the swapping of p1 and p2
   popt * pttemp; //pointer used for the swapping of pt1 and pt2

   // loop
   for (int i = 0; i < 2*n2steps; ++i)
   {

      bc(p1,pt1,bottemp,toptemp); //update boundary
      displace(p1,p2,pt1,pt2); // streaming from p1 to p2
#ifdef ACCELERATION
      epssin = eps*sin(omega*(((data_t) t[0])+0.5));
      collide(p2,pt2,epssin);
#else
      collide(p2,pt2); // collision and forces step
#endif
      // swap p1 p2 pointers, pt1, pt2 such that pt
      ptemp = p1;
      pttemp = pt1;
      p1 = p2;
      pt1 = pt2;
      p2 = ptemp;
      pt2 = pttemp;

      // increment time
      t[0]++;
   }
}

void simsteps(simulation * sim, int n2steps) { // step all the simulations
   for (int s = 0; s<NSIM; s++) {
      lonesteps(sim,s,n2steps);
   }
   calxyUvrhot(sim);
}

simulation * initfull() { // make 
   pop * p1, *p1f;
   pop * p2, *p2f;
   popt * pt1, *pt1f;
   popt * pt2, *pt2f;
   data_t *u,*v,*rho,*temperature;
   data_t *bottemp,*toptemp;
   unsigned long int * t;
   simulation * sim;
   sim = (simulation *) malloc(sizeof(simulation));//posix_memalign((void **) &sim, 4096, sizeof(simulation));
   //return sim;
   p1f = (pop *) malloc(NSIM*NXP2*NYP2*sizeof(pop));//posix_memalign((void **) &p1, 4096, NXP2*NYP2*sizeof(pop)); // or other p = (pop *)malloc(NXP2*NYP2*sizeof(pop)); method?
   p2f = (pop *) malloc(NSIM*NXP2*NYP2*sizeof(pop));// posix_memalign((void **) &p2, 4096, NXP2*NYP2*sizeof(pop));
   pt1f = (popt *) malloc(NSIM*NXP2*NYP2*sizeof(popt));//posix_memalign((void **) &pt1, 4096, NXP2*NYP2*sizeof(popt));
   pt2f = (popt *) malloc(NSIM*NXP2*NYP2*sizeof(popt));//posix_memalign((void **) &pt2, 4096, NXP2*NYP2*sizeof(popt));
   u = (data_t *) malloc(NSIM*NXP2*NYP2*sizeof(data_t));
   v = (data_t *) malloc(NSIM*NXP2*NYP2*sizeof(data_t));
   rho = (data_t *) malloc(NSIM*NXP2*NYP2*sizeof(data_t));
   temperature = (data_t *) malloc(NSIM*NXP2*NYP2*sizeof(data_t));
   t = (unsigned long int *) malloc(NSIM*sizeof(unsigned long int));

   toptemp = (data_t *) malloc(NSIM*NXP2*sizeof(data_t));
   bottemp = (data_t *) malloc(NSIM*NXP2*sizeof(data_t));
   
   sim->p1 = p1f;
   sim->p2 = p2f;
   sim->pt1 = pt1f;
   sim->pt2 = pt2f;
   sim->u = u;
   sim->v = v;
   sim->rho = rho;
   sim->temperature = temperature;
   sim->toptemp = toptemp;
   sim->bottemp = bottemp;
   sim->t = t;


   pop pzero;
   popt tzero;
   int pp,y,x,idx;
   data_t T;
   for (pp = 0; pp < NPOP; pp++)
      pzero.p[pp] = rho0/9.0;
   for (pp = 0; pp < NPOPT; pp++)
      tzero.p[pp] = T0/4.0;

   
   // intialize the simulations to a u=0, v=0 ,Temperature = T0 + small pertubation equalibria
   srand48((long) seed); // every simulation different initial conditions
   for (int s=0; s<NSIM;s++) {
      p1 = p1f+s*NXP2*NYP2;
      p2 = p2f+s*NXP2*NYP2;
      pt1 = pt1f+s*NXP2*NYP2;
      pt2 = pt2f+s*NXP2*NYP2;
      toptemp = sim->toptemp+s*NXP2;
      bottemp = sim->bottemp+s*NXP2;
      t = sim->t + s;
      t[0] = 0;
      for (y=0; y<NYP2; y++)
         for (x=0; x<NXP2; x++) {

            idx = IDX(y, x);
            p1[idx] = pzero;
            pt1[idx] = tzero;
            p2[idx] = pzero;
            pt2[idx] = tzero;

            //T = (T0+0.01*sin(2*M_PI*(x-1.)/(data_t)(NX-1.))); // inital temptature profile
            T = (T0+0.01*2.*(0.5-drand48())); // inital temptature profile
            pt1[idx].p[0] = 0.25*T*(1.+2.*vx(p1[idx]));
            pt1[idx].p[1] = 0.25*T*(1.+2.*vy(p1[idx]));
            pt1[idx].p[2] = 0.25*T*(1.-2.*vx(p1[idx]));
            pt1[idx].p[3] = 0.25*T*(1.-2.*vy(p1[idx]));
            pt2[idx].p[0] = 0.25*T*(1.+2.*vx(p2[idx]));
            pt2[idx].p[1] = 0.25*T*(1.+2.*vy(p2[idx]));
            pt2[idx].p[2] = 0.25*T*(1.-2.*vx(p2[idx]));
            pt2[idx].p[3] = 0.25*T*(1.-2.*vy(p2[idx]));
      }
      for (x=0;x<NXP2; x++) {
         toptemp[x] = TTOP0;
         bottemp[x] = TBOT0;
      }
   }

#ifdef ACCELERATION
   sim->eps = (data_t *) malloc(NSIM*sizeof(data_t));
   sim->omega = (data_t *) malloc(NSIM*sizeof(data_t));
   for (int si=0; si<NSIM;si++) { // memset?
      sim->eps[si] = 0.;
      sim->omega[si] = 0.;
   }
#endif

   calxyUvrhot(sim); // update u,v,rho,temperature for use in python
   
   return sim;
}

int testfun(int arg){
    // simulation * sim;
    // data_t *u;
    // u = (data_t *) malloc(sizeof(data_t)); //Hangs
    // printf("Hello"); //Hangs
    // sim = (simulation *) malloc(sizeof(simulation));//posix_memalign((void **) &sim, 4096, sizeof(simulation));
    return 5;
}


int main(int argc, char const *argv[]) // only for testing purposes (used in: make test)
{
   simulation * sim;
   printf("starting init\n");
   sim = initfull();
   printf("init done\n");
   printf("poiter to sim = %p\n", sim);
   printf("p1=%p,p2=%p,pt1=%p,pt2=%p\n", sim->p1,sim->p2,sim->pt1,sim->pt2);
   simsteps(sim,1000);
   calxyUvrhot(sim);
   printf("p1=%p,p2=%p,pt1=%p,pt2=%p\n", sim->p1,sim->p2,sim->pt1,sim->pt2);
   printf("%f\n",sim->p1[IDX(NY/2,NX/2)].p[0]);
   return 0;
}
