# see brother code of d2q9-cpu for better documentation on the setup/method/algorithm/make parameters.
# Here only the cuda implementation will be discussed
#
# This code has been written for the large part by Pinaki Kulmar (written: 2018)
# Where Gerben Beintema used/changed it to fit his needs for the thesis
# 
# There is a important difference to the gpu code:
# the population are split into multiple varibles thus copying them back to a single will result in a different order of dimentions
# for example the velocity populations are
# p(s,x,y,9) for cpu code
# p(9,s,x,y) for cuda code
# This is corrected for in the python code
# this makes the cuda code preform better but also makes it less readable
# 
# important functions of programs:
# * initfull return a simulation struct that holds all the varibles
# * simsteps(simulation *sim, int n2steps) 
#    - is the main function that steps all independed simulations for 2*n2steps number of lbm steps (needs to be even for pointer swapping)
#    - It copy the bottemp and toptemp first to device than steps 
#    -  and copies some of the results back (rho,u,v,temp,pt1,pt3)
# * calxyUvrhot(simulation *sim)
#    - copies the entire state back from the device to the host (and calculates rho,u,v,temp again to let them be up to date)
# * copystate(simulation *sim)
#    - copies the populations from the host to device, can for example be used to load from checkpoint
# 

#####################################################
#
# This Makefile works to create the shared libary object lbm{name}.so
#
#####################################################

# Gencode arguments (sm architecture : maxwell)
######################################################################
GENCODE_FLAGS ?= -gencode arch=compute_70,code=sm_70 #################
######################################################################

CUDA_PATH ?= "/usr/local/cuda/"

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

LIBRARIES := -lm

INCLUDES  := -I$(CUDA_PATH)/include

CUDA_LIBRARIES ?= -L$(CUDA_PATH)/lib64 -lcudart 

CUDA_FLAGS := -std=c++14

CPP_FLAGS := -std=c++14

SIM_FLAGS := -DNX=${NX} -DNY=${NY} -DNSIM=${NSIM} -DTBOT0=${TBOT0} -DTTOP0=${TTOP0} \
			-DT0=${T0} -D${wallprop} -Dalphag=${alphag} -Dff_body=${ff_body} -Dtau=${tau} \
			-Dtaut=${taut} -D${silent} -D${acceleration} -DDEVICE_ID=${device} -Dseednow=${seed} \
			-D${usedtype}


# Target Rules
all: build

lib: build

build: rlbm${name} lbm${name}.so

test: #run a simple test that only used temperature diffusion
	make clean
	make test.out NX=64 NY=64 NSIM=1 TBOT0=2. TTOP0=1. T0=1.5 wallprop=WALLED alphag=0. ff_body=0. \
	tau=0.5666 taut=0.5667 silent=NOTSILENT acceleration=NOTACCER device=0 seed=42 usedtype=usefloat
	./test.out
	make clean

rlbm${name}.o:main.c
	$(EXEC) $(HOST_COMPILER) $(SIM_FLAGS) -march=native $(INCLUDES) $(CPP_FLAGS) -o $@ -c $<

rlbm_gpu${name}.o:kernels.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(SIM_FLAGS) -m64 $(GENCODE_FLAGS) $(CUDA_FLAGS) -o $@ -c $<

rlbm${name}: rlbm${name}.o rlbm_gpu${name}.o
	$(EXEC) $(HOST_COMPILER) $(SIM_FLAGS) -o $@ $+ $(CUDA_LIBRARIES) $(LIBRARIES)

########

rlbmso${name}.o:main.c
	$(EXEC) $(HOST_COMPILER) $(SIM_FLAGS) -fPIC -march=native $(INCLUDES) $(CPP_FLAGS) -o $@ -c $<

rlbm_gpuso${name}.o:kernels.cu
	$(EXEC) $(NVCC) $(SIM_FLAGS) -Xcompiler -fPIC $(INCLUDES) -m64 $(GENCODE_FLAGS) $(CUDA_FLAGS) -o $@ -c $<

lbm${name}.so: rlbmso${name}.o rlbm_gpuso${name}.o
	$(EXEC) $(HOST_COMPILER) $(SIM_FLAGS) -o $@ $+ $(CUDA_LIBRARIES) $(LIBRARIES) -fPIC -shared

test.out: rlbmso${name}.o rlbm_gpuso${name}.o
	$(EXEC) $(HOST_COMPILER) $(SIM_FLAGS) -o $@ $+ $(CUDA_LIBRARIES) $(LIBRARIES) -fPIC

################
### cleaning ###
################

clean:
	rm -f rlbm rlbm_gpu${name}.o rlbm${name}.o rlbm${name}.so lbm${name}.so rlbmso${name}.o rlbm_gpuso${name}.o test${name}.out

superclean:
	rm -f rlbm* rlbm_gpu*.o rlbm*.o rlbm*.so lbm*.so rlbmso*.o rlbm_gpuso*.o test*.out

clobber: clean
