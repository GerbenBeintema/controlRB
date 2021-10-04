from controlRB.LBMd2q9ctypes import *
from matplotlib import pyplot as plt
import numpy as np

def get_timing(N=64,NSIM=1,nsteps=20000,mode='cpu'):
	lbm = lbmd2q9(mode=mode,NX=N,NY=N,NSIM=NSIM,verbose=False)
	timing = lbm.allsteps(nsteps=nsteps,updatexyUvrhot=False)
	lbm.clean()
	return timing

def scaling_steps(mode='cpu'):
	print('scalingsteps mode = ',mode)
	N = 64
	NSIM = 1
	try:
		for nsteps in [2,2*10**1,2*10**2,2*10**3,2*10**4]:
			timing = gettiming(N=N,NSIM=NSIM,nsteps=nsteps,mode=mode)
			speed = timing/(nsteps/100)
			print("{:6d} nsteps gives {:.2E} sec with {:.2E} sec/(100 nsteps)".format(nsteps,timing,speed))
	except KeyboardInterrupt:
		print('KeyboardInterrupt')

def scaling_NSIM(mode='cpu'):
	print('simscaling mode = ',mode)
	N = 64
	nsteps = 4*10**2
	try:
		for NSIM in [1,2,5,10,20,50,100,200,500,1000,2000]:
			timing = gettiming(N=N,NSIM=NSIM,nsteps=nsteps,mode=mode)
			speed = timing/(NSIM)/(nsteps/100)
			print("{:6d} simulations gives {:.2E} sec with {:.2E} sec/(NSIM 100 nsteps)".format(NSIM,timing,speed))
	except KeyboardInterrupt:
		print('KeyboardInterrupt')

def scaling_size(mode='cpu'):
	print('sizescaling mode = ',mode)
	nsteps = 4*10**2
	NSIM = 1
	try:
		for N in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
			timing = gettiming(N=N,NSIM=NSIM,nsteps=nsteps,mode=mode)
			speed = timing/(N*N/(64**2)*(nsteps/100))
			print("{:5d}**2 simulation size gives {:.2E} sec with {:.2E} sec/((N*N)/64**2 100 nsteps)".format(N,timing,speed))
	except KeyboardInterrupt:
		print('KeyboardInterrupt')

if __name__=='__main__':
	scaling_steps(mode='cpu')
	scaling_NSIM(mode='cpu')
	scaling_size(mode='cpu')
