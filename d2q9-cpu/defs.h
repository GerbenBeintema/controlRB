// see Makefile for documentation

#ifndef NX
#define NX 64 // domian size horizontal
#endif
#ifndef NY
#define NY 64 // domian size vertical
#endif

#ifndef NSIM
#define NSIM 1 // number of independed simulations
#endif

#define NXP2 (NX+2)
#define NYP2 (NY+2)
#ifndef seed
#define seed 42
#endif

#define NPOP 9
#define NPOPT 4

#ifndef tau
#define tau 0.5666 // time constant velocity nu = cs2*(tau-0.5)
#endif
#ifndef taut
#define taut 0.5666 // time constant tempature kappa = 0.25*(0.5*taut-1.0) = 0.5*(taugt
#endif
#define tauR (1.-1./(2.*tau))
#ifndef alphag
#warning warning: setting alphag to zero
#define alphag (0.0)
#endif
#ifndef ff_body
#warning warning: setting ff_body to zero
#define ff_body (0.0)
#endif

#define cs2  (1.0 /  3.0) // speed of sound
#define cs22 (2.0 *  cs2)
#define cssq (2.0 /  9.0)
#define cs2i (3.0)

#define rt0  (4.0 /  9.0) // weights
#define rt1  (1.0 /  9.0)
#define rt2  (1.0 / 36.0)

/* WARNING vx is rho.vx */
#define vx(a) (a.p[1]+a.p[5]+a.p[8]-a.p[3]-a.p[6]-a.p[7]) // velocity x 
#define vy(a) (a.p[2]+a.p[5]+a.p[6]-a.p[4]-a.p[7]-a.p[8]) // velocity y
#define vt(a) (a.p[0]+a.p[1]+a.p[2]+a.p[3]) // tempature
#define  m(a) (a.p[0]+a.p[1]+a.p[2]+a.p[3]+a.p[4]+a.p[5]+a.p[6]+a.p[7]+a.p[8]) // mass

//#define IDX(i,k) (NXP2*(i) + (k)) // old one with (y,x) as dimentions
#define IDX(j,i) ((j) + NYP2*(i))


#ifndef TTOP0
#define TTOP0 (1.)
#endif
#ifndef TBOT0
#define TBOT0 (2.)
#endif
#ifndef T0
#define T0    (0.5*(TTOP0+TBOT0))
#endif


#ifdef usefloat
	#define data_t float
	//#warning "Using float"
#else
	#define data_t double
	//#warning "Using double"
#endif

#define rho0 1.0

//#define PERIODIC
//#define WALLED
#ifndef WALLED //&?
#ifndef PERIODIC
# error ERROR: NO WALL PROP SET !!!
#endif
#endif