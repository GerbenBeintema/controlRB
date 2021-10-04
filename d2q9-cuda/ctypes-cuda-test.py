from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# python file to test if cuda shared library is working, this will
# 1. compile
# 2. run a test function on cpu
# 3. init cuda simualtion
# 4. 
import numpy as np
from subprocess import call

import ctypes

NX = 64
NY = 32
NSIM = 8
NYP2 = NY+2
NXP2 = NX+2
Nf = NXP2*NYP2*NSIM
NPOP = 9
NPOPT = 4
seed = 42

tau = 0.5666
taut = 0.5666
tauR = (1.-1./(2.*tau))
alphag = 0.000000
ff_body = 0.
TTOP0 = 1.
TBOT0 = 2.
T0 = 0.5*(TTOP0+TBOT0)
rho0 = 1.0
wallprop = "WALLED"

# only used interntally in the cuda code thus not needed

# class cudaVars(ctypes.Structure):
# 	_fields_ = [] #maybe need to add, test without -> can be done without
# class hostVars(ctypes.Structure):
# 	_fields_ = [('host_u', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('host_v', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('host_rho', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('host_temperature', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('host_bottemp',ctypes.POINTER(ctypes.c_double*(NXP2*NSIM))),
# 				('host_toptemp',ctypes.POINTER(ctypes.c_double*(NXP2*NSIM))),
# 				('h_f0ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_f1ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_f2ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_f3ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_f4ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_f5ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_f6ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_f7ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_f8ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_ft0ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_ft1ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_ft2ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_ft3ref', ctypes.POINTER(ctypes.c_double*Nf)),
# 				('h_p1ref', ctypes.POINTER(ctypes.c_double*(Nf*NPOP))),
# 				('h_pt1ref', ctypes.POINTER(ctypes.c_double*(Nf*NPOPT)))
# 	           ]


# class gpuVars(ctypes.Structure):
# 	_fields_ = [('cv',ctypes.POINTER(cudaVars)),
# 	            ('hv',ctypes.POINTER(hostVars)),
# 	            ('threads_dispColl',ctypes.c_uint),
#                 ('threads_bcTB',ctypes.c_uint),
#                 ('threads_bcEW',ctypes.c_uint),
#                 ('blocks_dispColl',ctypes.c_uint),
#                 ('blocks_bcTB',ctypes.c_uint),
#                 ('blocks_bcEW',ctypes.c_uint)
#                 ]
class simulation(ctypes.Structure):
    """docstring for simulation"""
    _fields_ = [('p1',ctypes.POINTER(ctypes.c_double*(Nf*NPOP))), 
                ('p2',ctypes.POINTER(ctypes.c_double*(Nf*NPOP))), #just a dummy, never copied back from device
                ('pt1',ctypes.POINTER(ctypes.c_double*(Nf*NPOPT))), 
                ('pt2',ctypes.POINTER(ctypes.c_double*(Nf*NPOPT))), #just a dummy, never copied back from device
                ('u', ctypes.POINTER(ctypes.c_double*(Nf))),
                ('v', ctypes.POINTER(ctypes.c_double*(Nf))),
                ('rho', ctypes.POINTER(ctypes.c_double*(Nf))),
                ('temperature', ctypes.POINTER(ctypes.c_double*(Nf))),
                ('bottemp', ctypes.POINTER(ctypes.c_double*(NSIM*NXP2))),
                ('toptemp', ctypes.POINTER(ctypes.c_double*(NSIM*NXP2))),
                ('gpu', ctypes.c_void_p)
                #('gpu',ctypes.POINTER(gpuVars))
                ]



def wrap_function(funcname, restype=None, argtypes=None):
	"""Simplify wrapping ctypes functions"""
	func = lbmlib.__getattr__(funcname)
	func.restype = restype
	func.argtypes = argtypes #needs to be a list or None
	return func


#make lib
call(['make','clean'])
call(['make','lib',
      'NX='+str(NX),
      'NY='+str(NY),
      'NSIM='+str(NSIM),
      'TBOT0='+str(TBOT0),
      'TTOP0='+str(TTOP0),
      'T0='+str(T0),
      'wallprop='+str(wallprop),
      'alphag='+str(alphag),
      'ff_body='+str(ff_body),
      'tau='+str(tau),
      'taut='+str(taut),
      'device='+str(0),
      'usedtype='+str('usedouble'),
      'seed='+str(42),
      'acceleration=NOTACCELERATION',
      'silent=NOTSILENT'])
lbmlib = ctypes.CDLL('./lbm.so')


### basic shared libary
print('python: running function to see if shared libary is working...')
ctypes_test_that_cals_x_time_x = wrap_function('ctypes_test_that_cals_x_time_x',restype=ctypes.c_int, argtypes = [ctypes.c_int])
print('python: 2*2=',ctypes_test_that_cals_x_time_x(2))
print('done test fun')

### init cuda simulation
print('starting init')
initfun = wrap_function('initfull', restype = ctypes.POINTER(simulation))
simpointer = initfun()

p1pointer = simpointer.contents.p1
p2pointer = simpointer.contents.p2
pt1pointer = simpointer.contents.pt1
pt2pointer = simpointer.contents.pt2
upointer = simpointer.contents.u
vpointer = simpointer.contents.v
rhopointer = simpointer.contents.rho
temppointer = simpointer.contents.temperature
bottemppointer = simpointer.contents.bottemp
toptemppointer = simpointer.contents.toptemp


usource = np.ctypeslib.as_array(upointer.contents).reshape((NSIM,NXP2,NYP2))
vsource = np.ctypeslib.as_array(vpointer.contents).reshape((NSIM,NXP2,NYP2))
rhosource = np.ctypeslib.as_array(rhopointer.contents).reshape((NSIM,NXP2,NYP2))
tempsource = np.ctypeslib.as_array(temppointer.contents).reshape((NSIM,NXP2,NYP2))
#p1source = np.ctypeslib.as_array(p1pointer.contents).reshape((NSIM,NXP2,NYP2,NPOP))
#pt1source = np.ctypeslib.as_array(pt1pointer.contents).reshape((NSIM,NXP2,NYP2,NPOPT))
#p2source = np.ctypeslib.as_array(p2pointer.contents).reshape((NSIM,NXP2,NYP2,NPOP))
#pt2source = np.ctypeslib.as_array(pt2pointer.contents).reshape((NSIM,NXP2,NYP2,NPOPT))
bottemp = np.ctypeslib.as_array(bottemppointer.contents).reshape((NSIM,NXP2))[:,1:NX+1]
toptemp = np.ctypeslib.as_array(toptemppointer.contents).reshape((NSIM,NXP2))[:,1:NX+1]


flipcrop = lambda x: np.swapaxes(np.flip(x[:,1:NX+1,1:NY+1],axis=2),1,2) #this makes plotting and visualisation more easy
#p = flipcrop(p1source)
#pt = flipcrop(pt1source)

u = flipcrop(usource)
v = flipcrop(vsource)
rho = flipcrop(rhosource)
temp = flipcrop(tempsource)

stepsfun = wrap_function('simsteps',restype=None,argtypes=[ctypes.POINTER(simulation),ctypes.c_int])
calxyUvrhot = wrap_function('calxyUvrhot',restype=None,argtypes=[ctypes.POINTER(simulation)])

host_temperature = tempsource
print("host_temperature.shape",host_temperature.shape)
print("y_profile",host_temperature[0,int(NXP2/2),:])

simsteps = wrap_function('simsteps',restype=None, argtypes = [ctypes.POINTER(simulation),ctypes.c_int])
for i in range(10):
	simsteps(simpointer,ctypes.c_int(500));

	print(i,"y_profile",host_temperature[0,int(NXP2/2),:])

print('this should be approximatlly a linear profile going from the bottom (T=2) to the top (T=1)\n\n')


temp = host_temperature[0,int(NXP2/2),1:-1]
tlin = np.linspace(2,1,num=NY+1)
tlin = (tlin[1:]+tlin[:-1])/2
print('temp - linear profile=')
print(temp-tlin)
print('diff=',np.mean((temp-tlin)**2)**0.5,'= approx 0.0011966155317232711')

"""
print('dummy array range(5)**2=',dummy)
print('dir gpu:',dir(gpu.contents))
print('threads_hallo uint=',gpu.contents.threads_hello)
hostvectest =  np.ctypeslib.as_array(gpu.contents.hv.contents.host_vec.contents)
print('host_vec=',hostvectest)

callGPUStepper = wrap_function('callGPUStepper',argtypes=[ctypes.POINTER(gpuVars)])
print('stepper function...')
callGPUStepper(gpu)
print('stepper done')
print('host vec *= 2 =')
print(hostvectest)

"""
print('done with test, Have a nice day :D')