
# This the code by Gerben Beintema (id: 0907710) for his master thesis of AP 2018-2019, 
# Thesis: "Finding noval control statagies for flow using reinforcement learning and high performance simulations"
# Contact g.i.beintema@student.tue.nl and gerben.beintema@gmail.com for more information
# 
# This code has been used as a template for the creation of its sister code d2q9-cuda 
# that implements the same problem but than in cuda
# 
# The code has been setup to be able to be loaded by python as a shared libary (just as the cuda code)
# This has made the connections to control algorithms easier. 
#
# This is lbm code for a 2D with 9 velocity population and 4 temperature populations 
# with and extra dimention for multiple simulations
# and usable interface to python where the temperature of the boundary can be dynemicly changed.
# For details see papers (or contact me by mail)
# 1. 6.4 Alternative Forcing Schemes (p. 241) method: He et al. [39]
# 2. Numerical study of lattice Boltzmann methods for a convection–diffusion equation coupled with Navier–Stokes equations


### SIMULATION SPACE ####
#
#
# dimentions in simulation are NSIM, NXP2 (NX + 2), NYP2 (NY + 2),... (first s, x than y)
# the ith simuation dimentions is denoted by a 's' 
# thus for example u(s,x,y) has the size of (NSIM,NXP2,NYP2) 
#
#                            Top boundary (no slip tempature controlable)
#                          0,NY+1.............. NX+1,NY+1
#y                         .                  .
#^                         .                  .
#|noslip-wall (or periodic).      domain      . noslip-wall (or periodic)
#|                         .                  .
#|                         .                  .
#|                         0,0.................0,NX+1
#|                           bottom boundary (no slip tempature controlable)
#+-----> x
# with NSIM being a the first dimintion 
#

 
# Compile Parameters
# * NX
#	- Number of lbm notes in the width of the domain
# * NY
#	- Number of lbm notes in the height of the domain
# * NSIM
#  	- Number of independent simulations of the 2D rayleight benard system you want to run concurrently
# * TBOT0
#   - Initial bottom temperature (default is 2.)
# * TTOP0
#   - Initial bottom temperature (default is 1.)
# * T0
#   - Initial temperature domain (default is (0.5*(TTOP0+TBOT0)))
# * wallprop
#   - The property you want for your wall. no default with 
#      1. wallprop=WALLED -> No slip walls on both sides
#      2. wallprop=PERIODIC -> Periodic boundaries 
# * alphag
#   - bouency body force coefficent (alphag = alpha * g = (expantion coefficent)*(gravitation acceraltion))
# * ff_body
#   - uniform body force in the horizontal direction (having periodic boundaries is advised), only used for validation in my thesis
# * tau
#   - time constant of the velocity (p) population (default: 0.5666) determining the viscosity, nu = cs2*(tau-0.5) = 0.333*(tau-0.5)
# * taut
#   - time constant of the temperature (pt)
# * acceleration
#   - Enables vertical acceraltions (changing the gravity) of the cell. Used in gravity modulation stabilisation
# * seed
#   - seed
# * usedtype
#   - chose the default data type: usefloat or usedouble, using doubles is faster on cpu but float is faster on GPU
# * name
#   - name of compiled (no spaces and such) such that multiple instances can be compiled at the same time and they do not overwrite eachother.




lib:
	gcc -Wall -lm -fPIC -shared \
	-DNX=${NX} -DNY=${NY} -DNSIM=${NSIM} -DTBOT0=${TBOT0} -DTTOP0=${TTOP0} -DT0=${T0} -D${wallprop} \
	-Dalphag=${alphag} -Dff_body=${ff_body} -Dtau=${tau} -Dtaut=${taut} -D${acceleration} -Dseed=${seed} \
	-D${usedtype} \
	-o lbm${name}.so lbmd2q9.c -lc

libsimp:
	gcc -Wall -lm -fPIC -shared  -DNX=64 -DNY=64 -DNSIM=8 -DTBOT0=2. -DTTOP0=1. -DWALLED -Dalphag=0.00001 -Dff_body=0. -o lbm.so lbmd2q9.c -lc

clean:
	rm -f lbm${name}.so  a.out 

superclean:
	rm -f lbm*.so  a.out analysis.out

test:
	rm -f lbm.so a.out
	gcc -Wall -DNX=64 -DNY=64 -DNSIM=2 -DWALLED -Dalphag=0.00001 -Dff_body=0. -Dusedouble -o a.out lbmd2q9.c -lm -lc  
	./a.out
	rm ./a.out

profile: #profile the code
	rm -f lbm.so a.out
	gcc -Wall -pg -DNX=64 -DNY=64 -DNSIM=2 -DWALLED -Dalphag=0.00001 -Dff_body=0. -Dusedouble -o TestGprof lbmd2q9.c -lm -lc 
	./TestGprof
	gprof -b TestGprof gmon.out > analysis.out
	rm -f TestGprof gmon.out
