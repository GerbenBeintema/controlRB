from controlRB.LBMd2q9ctypes import *
from controlRB.validation import *
from matplotlib import pyplot as plt
import numpy as np

#This has some function that validate that the cuda and cpu code are equvilent

def cpucuda_compare_with_p_pt():
	print('\n\n starting cpucuda_compare_with_p_pt \n\n')
	print('cpu version:')
	lbm = lbmd2q9(mode='cpu')
	lbm.allsteps(nsteps=4000)
	print('lbm.p1source.shape:',lbm.p1source.shape)
	print('np.sum(lbm.p1source):',np.sum(lbm.p1source))
	p1cpu = np.copy(lbm.p1source)
	pt1cpu = np.copy(lbm.pt1source)

	print('cuda verion:')
	lbm = lbmd2q9(mode='cuda')
	lbm.allsteps(nsteps=4000)
	print('lbm.p1sourcecuda.shape:',lbm.p1sourcecuda.shape)
	print('lbm.p1source.shape:',lbm.p1source.shape)
	print('np.sum(lbm.p1source):',np.sum(lbm.p1source))
	p1cuda = np.copy(lbm.p1source)
	pt1cuda = np.copy(lbm.pt1source)
	diff = np.abs(p1cuda-p1cpu)
	print('error of np.max(np.abs(p1cuda-p1cpu)):',np.max(diff))
	print('argmax(np.abs(p1cuda-p1cpu))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	diff = np.abs(pt1cuda-pt1cpu)
	print('error of np.max(np.abs(pt1cuda-pt1cpu)):',np.max(diff))
	print('argmax(np.abs(pt1cuda-pt1cpu))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	print('\n\n cpucuda_compare_with_p_pt passed :D \n\n')

def cpucuda_compare_with_thermaldiff():
	print('\n\n starting cpucuda_compare_with_thermaldiff \n\n')
	lbm = validate_thermaldiff(file=True,mode='cpu')
	p1cpu = np.copy(lbm.p1source)
	pt1cpu = np.copy(lbm.pt1source)
	lbm = validate_thermaldiff(file=True,mode='cuda')
	p1cuda = np.copy(lbm.p1source)
	pt1cuda = np.copy(lbm.pt1source)
	diff = np.abs(p1cuda-p1cpu)
	print('error of np.max(np.abs(p1cuda-p1cpu)):',np.max(diff))
	print('argmax(np.abs(p1cuda-p1cpu))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	diff = np.abs(pt1cuda-pt1cpu)
	print('error of np.max(np.abs(pt1cuda-pt1cpu)):',np.max(diff))
	print('argmax(np.abs(pt1cuda-pt1cpu))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	print('\n\n cpucuda_compare_with_thermaldiff passed :D \n\n')

def cpucuda_compare_with_Poiseuille(plot=False):
	print('\n\n starting cpucuda_compare_with_Poiseuille \n\n')
	lbm = validate_Poiseuille(file=True,mode='cpu')
	p1cpu = np.copy(lbm.p1source)
	pt1cpu = np.copy(lbm.pt1source)
	rhocpu = np.copy(lbm.rho)
	

	diff = np.abs(lbm.rho-np.sum(lbm.p,axis=-1))
	print('error of np.max(lbm.rho-np.sum(lbm.p,axis=-1)):',np.max(diff))
	print('argmax(np.abs(lbm.rho-np.sum(lbm.p,axis=-1)))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	diff = np.abs(lbm.temp-np.sum(lbm.pt,axis=-1))
	print('error of np.max(np.abs(lbm.temp-np.sum(lbm.pt,axis=-1))):',np.max(diff))
	print('argmax(np.abs(lbm.temp-np.sum(lbm.pt,axis=-1)))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	p = np.copy(lbm.p[0])
	uwithp = (p[:,:,1]+p[:,:,5]+p[:,:,8]-(p[:,:,3]+p[:,:,7]+p[:,:,6]))/rhocpu
	ucpu = np.copy(lbm.u)
	diff = np.abs(uwithp-ucpu)
	print('error of np.max(np.abs(uwithp-ucpu)):',np.max(diff))
	print('argmax(np.abs(uwithp-ucpu))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	
	print('\ncuda..\n\n')
	lbm = validate_Poiseuille(file=True,mode='cuda')
	p1cuda = np.copy(lbm.p1source)
	pt1cuda = np.copy(lbm.pt1source)
	diff = np.abs(lbm.rho-np.sum(lbm.p,axis=-1))
	rhocuda = np.copy(lbm.rho)
	print('error of np.max(lbm.rho-np.sum(lbm.p,axis=-1)):',np.max(diff))
	print('argmax(np.abs(lbm.rho-np.sum(lbm.p,axis=-1)))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	diff = np.abs(lbm.temp-np.sum(lbm.pt,axis=-1))
	print('error of np.max(np.abs(lbm.temp-np.sum(lbm.pt,axis=-1))):',np.max(diff))
	print('argmax(np.abs(lbm.temp-np.sum(lbm.pt,axis=-1)))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	diff = np.abs(rhocuda-rhocpu)
	print('error of np.max(np.abs(rhocuda-rhocpu)):',np.max(diff))
	print('argmax(np.abs(rhocuda-rhocpu))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10

	
	ucuda = np.copy(lbm.u)
	diff = np.abs(p1cuda-p1cpu)
	print('error of np.max(np.abs(p1cuda-p1cpu)):',np.max(diff))
	print('argmax(np.abs(p1cuda-p1cpu))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	diff = np.abs(pt1cuda-pt1cpu)
	print('error of np.max(np.abs(pt1cuda-pt1cpu)):',np.max(diff))
	print('argmax(np.abs(pt1cuda-pt1cpu))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	p = np.copy(lbm.p[0])
	uwithp = (p[:,:,1]+p[:,:,5]+p[:,:,8]-(p[:,:,3]+p[:,:,7]+p[:,:,6]))/rhocuda
	ucuda = np.copy(lbm.u)
	diff = np.abs(uwithp-ucuda)
	print('error of np.max(np.abs(uwithp-ucuda)):',np.max(diff))
	print('argmax(np.abs(uwithp-ucuda))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10

	diff = np.abs(ucpu-ucuda)
	print('np.max(ucpu)',np.max(ucpu))
	if plot:
		plt.imshow(diff[0])
		plt.colorbar()
		plt.savefig('udiff.jpg')
		plt.close()
	print('error of np.max(np.abs(ucpu-ucuda)):',np.max(diff))
	print('argmax(np.abs(ucpu-ucuda))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	print('\n\n cpucuda_compare_with_Poiseuille passed :D \n\n')

def cpucuda_compare_with_Ra1e5():
	print('\n\n starting cpucuda_compare_with_Ra1e5 \n\n')
	lbm = validate_Ra1e5(mode='cpu',file=True)
	p1cpu = np.copy(lbm.p1source)
	lbm = validate_Ra1e5(mode='cuda',file=True)
	p1cuda = np.copy(lbm.p1source)
	diff = np.abs(p1cuda-p1cpu)
	print('error of np.max(np.abs(p1cuda-p1cpu)):',np.max(diff))
	print('argmax(np.abs(p1cuda-p1cpu))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	print('\n\n cpucuda_compare_with_Ra1e5 passed :D \n\n')

def cpucuda_compare_with_acceleration_modulation():
	print('\n\n starting cpucuda_compare_with_acceleration_modulation \n\n')
	lbm = lbmd2q9(mode='cpu',accelerationcontrol=True,eps0=1,omega0=3.14*2/(1000))
	lbm.allsteps(nsteps=1000)
	pcpu = np.copy(lbm.p)
	lbm.clean()
	lbm = lbmd2q9(mode='cuda',accelerationcontrol=True,eps0=1,omega0=3.14*2/(1000))
	lbm.allsteps(nsteps=1000)
	pcuda = np.copy(lbm.p)
	lbm.clean()
	diff = np.abs(pcpu-pcuda)
	print('error of np.max(np.abs(pcpu-pcuda)):',np.max(diff))
	print('argmax(np.abs(pcpu-pcuda))',np.unravel_index(np.argmax(diff, axis=None), diff.shape))
	assert np.max(diff)<10**-10
	print('\n\n cpucuda_compare_with_acceleration_modulation passed :D \n\n')


if __name__=='__main__':
	cpucuda_compare_with_p_pt()

	cpucuda_compare_with_acceleration_modulation()

	cpucuda_compare_with_thermaldiff()

	cpucuda_compare_with_Poiseuille()

	cpucuda_compare_with_Ra1e5()
