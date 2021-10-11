# this is the shared liberary 




### SIMULATION SPACE ####
# dimentions in simulation are NSIM,NXP2,NYP2,... (first x than y)
#            Top boundary (no slip tempature controlable)
#y      0,NY+1...... NX+1,NY+1
#^     .              .
#|wall .              . wall (or periodic)
#|     .              .
#|     0,0 ........ 0,NX+1
#|          bottom boundary (no slip tempature controlable)
#+-----> x

### PYTHON SPACE ###
# remove ghost nodes, flip y-axis, swaps y-axis with x-axis: thus dimentions become NSIM,NY,NX,... (first y than x)
# +-----------> x 
# |      Top boundary (no slip tempature controlable)
# |     0,0 ........ 0,NX-1
# |     .             .
# |wall .             . wall (or periodic)
# v     .             .
# y     NY-1,0 ......NY-1,NX-1
#        bottom boundary (no slip tempature controlable)



from __future__ import absolute_import,division,print_function
import os #dirs
from os import path
import ctypes #load library lbm.so
from subprocess import call #used to make lib
import numpy as np
import time #timing of functions
try:
    from matplotlib import pyplot as plt
    try:
        plt.plot([0,1]) #this will create RuntimeError error on volta
    except RuntimeError:  #plt.switch_backend('agg') #need to be active for volta plotting
        plt.switch_backend('agg') #this does not work on my laptop thus I made this weird construct
                                  #better solution would be preferred.
    plt.close()
    import matplotlib.cbook
    import warnings
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    import matplotlib.animation as animation #cannot be used on volta, needs some sudo modules
except ImportError:
    print('failed to load matplotlib')
try:
    from IPython.display import HTML #HTML videos
    HTMLloaded = True
except ImportError:
    HTMLloaded = False


NPOP = 9 # lbm velocity populations
NPOPT = 4 # temperature populations

try:
    dlclose_func = ctypes.CDLL(None).dlclose #for deloading library, does not work on windows, fix that later
    dlclose_func.argtypes = [ctypes.c_void_p]
    dlclose_func.restype = ctypes.c_int
    dlclose_active = True
except:
    dlclose_active = False
    print('deloading function not able to be load (are you running on windows?), watch out when loading multiple instances of lbmd2q9')
toppath = path.join(path.dirname(__file__),'..')


def makelib(self):
    """
    making the libary only when structure is: 
    # |- controlRB (where this file is located)
    # |- d2q9-cuda
    # |- d2q9-cpu

    """
    stardir = path.abspath('.')
    os.chdir(path.join(toppath,self.libdir))
    arg = ['make','clean','name='+str(self.name)]
    if self.verbose:
        call(arg)
    else:
        call(arg+['--silent'])
    if self.verbose: print('making lbm.so...')
    arg = ['make','lib',
          'NX='+str(self.NX),
          'NY='+str(self.NY),
          'NSIM='+str(self.NSIM),
          'TBOT0='+str(self.TBOT0),
          'TTOP0='+str(self.TTOP0),
          'T0='+str(self.T0),
          'wallprop='+str(self.wallprop),
          'alphag='+str(self.alphag),
          'ff_body='+str(self.ff_body),
          'tau='+str(self.tau),
          'taut='+str(self.taut),
          'device='+str(self.device),
          'seed='+str(self.seed),
          'usedtype='+str(self.usedtype),
          'name='+str(self.name)]
    if self.accelerationcontrol:
        arg += ['acceleration=ACCELERATION']
    else:
        arg += ['acceleration=NOTACCELERATION'] #plz
    if not self.verbose:
        arg += ['silent=SILENT']+['--silent']
    else:
        arg += ['silent=NOTSILENT']
    call(arg)
    os.chdir(stardir) #

deload_warned = False
def loadlib(self): #add options
    """load libary by 
     * first deloading it (this is needed if any of the parameters are changed such as alphag, tau, NX, ect)
     * than making the libarary at self.libdir location
     * finally loading it with CDLL"""
    # global eval('lbmlib{}'.format(self.name)) #only able to load 1 module at the same time
    # if 'lbmlib{}'.format(self.name) in globals(): #only true when lbmlib is already used than deloads it.
    #     if self.verbose: print('deloading old...')
    global libs #a list of all the lbm libraries loaded [(libname,lib),...]
    libname = 'lbmlib{}'.format(self.name)
    if 'libs' not in globals():
        if self.verbose: print('making new libs...')
        libs = []
    if libname in [l[0] for l in libs]: #only true when lbmlib is already used than deloads it.
        if self.verbose: print('deloading old...')
        ithlib = [i for i,l in enumerate(libs) if l[0]==libname][0]
        if dlclose_active: 
            dlclose_func(libs[ithlib][1]._handle)
        else:
            global deload_warned
            if deload_warned==False:
                warnings.warn("deloading was not able to deload but you tried to deload a by compiling with the same name = '{}'...".format(libname))
                deload_warned = True
        makelib(self)
        lbmlib = ctypes.CDLL(path.join(toppath,self.libdir)+'/lbm{}.so'.format(self.name))
        libs[ithlib][1] = lbmlib
    else:
        makelib(self)
        lbmlib = ctypes.CDLL(path.join(toppath,self.libdir)+'/lbm{}.so'.format(self.name))
        libs.append([libname,lbmlib])
    return lbmlib
    
def wrap_function(funcname, lib, restype=None, argtypes=None):
    """Simplify wrapping ctypes functions, helps a lot with catching errors."""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

class lbmd2q9(object):
    '''The class holder with d2q9 lbm for thermal convection 
        with multiple simulation with boundary control

        Input of lbmd2q9:
            mode: "cpu" for "cuda"
                Choses the opperation mode for eather 
                "cpu": d2q9-cpu
                "cuda": d2q9-cuda
            NX,NY: domain size
                the number of real nodes in the domain
            NSIM: Number of parallel simulations
            TBOT0,TTOP0: Initial uniform Temparature bottom/top temperature boundary
            T0: Starting Temparature fluid 
                Default = (TBOT0+TTOP0)/2
            wallprop: "Walled" or "PERIODIC"
            Ra: rayleigh number
            ff_body: constant body force in x direction
            Pr: Prandtl Number = nu/kappa
            verbose: Run in verbose (False will make it silent)

        Usage example:
            lbm = lbmd2q9() #64 by 64 at Ra=1e5 in cpu mode
            lbm.allsteps(nsteps=1000) #1000 simulation steps
            lbm.flowtemplot() #shows plot of temperature with flow field
            lbm.allsteps(nsteps=1000,bottemp=3) #sets temparture of bottom plate to 3 in all simulations
            lbm.tempplot() #shows temperature plot
            lbm.allsteps(nsteps=1000,bottemp=[2+np.sin(lbm.x/lbm.NX*np.pi*2)]) 
            #sets temparture of bottom plate 2 plus a sin

        Docs:
            TODO
                '''
    def __init__(self,mode='cpu',
                 NX=64,NY=64,NSIM=1,TBOT0=2.,TTOP0=1.,T0=None,wallprop='WALLED',
                 Ra=1e5,ff_body=0.,Pr = 0.71, tau=0.5666, verbose=True,seed=42,usedtype='double',
                 accelerationcontrol=False,eps0=0.,omega0=0.,name='',
                 device=None):
        '''The class holder with d2q9 lbm for thermal convection 
            with multiple simulation with boundary control

            Input:
                mode: "cpu" for "cuda"
                    Choses the opperation mode for eather 
                    "cpu": d2q9-cpu
                    "cuda": d2q9-cuda
                NX,NY: domain size
                    the number of real nodes in the domain
                NSIM: Number of parallel simulations
                TBOT0,TTOP0: Initial uniform Temparature bottom/top temperature boundary
                T0: Starting Temparature fluid 
                    Default = (TBOT0+TTOP0)/2
                wallprop: "Walled" or "PERIODIC"
                Ra: rayleigh number
                ff_body: constant body force in x direction
                Pr: Prandtl Number = nu/kappa
                verbose: Run in verbose (False will make it silent)
                name: a valid variable name (only letter and numbers) for the lib that will be generated'''
        self.verbose = verbose
        if verbose: print('mode = '+mode)
        assert mode=='cpu' or mode=='cuda', 'only cpu mode or cuda mode'
        self.mode = mode
        assert usedtype=='double' or usedtype=='float', '{}'.format(usedtype)
        if accelerationcontrol and mode=='cuda' and usedtype=='float':
            print('\n\n\nThere is a weird bug that causes the time not be copied to put to GPU when accelerationcontrol+cuda+float is active, Be warned \n\n\n')
        self.usedtype = 'use'+usedtype
        self.name=name
        if self.verbose: print(name)
        if self.mode=='cuda':
            if device is None:
                import os
                import gpustat
                import random

                stats = gpustat.GPUStatCollection.new_query()
                if len(stats)==4:
                    stats = stats[2:] #temp, restrict to 2 and 3 for pinaki
                ids = map(lambda gpu: int(gpu.entry['index']), stats)
                ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
                pairs = list(zip(ids, ratios))
                random.shuffle(pairs)
                bestGPU = min(pairs, key=lambda x: x[1])[0]

                if self.verbose: print("in LBMd2q9ctypes setGPU: Setting GPU to: {}".format(bestGPU))
                os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
                self.device = 0 #setGPU sets the best gpu to id=0
            else:
                self.device = device
            self.libdir = 'd2q9-cuda'
            if self.verbose: print('selected {} gpu'.format(self.device))
        elif self.mode=='cpu':
            self.device = 0 #does not do anything in cpu mode
            self.libdir = 'd2q9-cpu'
        else:
            assert False, 'Mode need to be "cpu" for "cuda"'

        self.Pr = Pr
        self.tau = tau
        self.taut = (self.tau-0.5)/Pr+0.5
        self.cs2 = 1./3.
        self.nu = self.cs2*(self.tau-0.5)
        self.kappa = 0.25*(2*self.taut-1) #https://www.math.nyu.edu/~billbao/report930.pdf

        
        self.NX = NX
        self.NY = NY
        self.seed = seed

        self.TBOT0 = float(TBOT0)
        self.TTOP0 = float(TTOP0)
        if T0 is None:
            self.T0 = (self.TBOT0+self.TTOP0)/2
        else:
            self.T0 = float(T0)
        self.Ra = float(Ra)
        self.accelerationcontrol = accelerationcontrol
        assert wallprop == 'WALLED' or wallprop == 'PERIODIC'
        self.wallprop = wallprop
        self.alphag = self.Ra*(self.kappa*self.nu)/((self.TBOT0-self.TTOP0)*self.NY**3) if self.Ra!=0. else 0.
        self.ff_body = float(ff_body) #body force scaling with density
        NYP2 = NY+2
        NXP2 = NX+2
        self.NXP2 = NXP2
        self.NYP2 = NYP2
        self.NSIM = NSIM
        self.Nf = self.NXP2*self.NYP2*self.NSIM
        Nf = self.Nf
        c_type = ctypes.c_double if self.usedtype=='usedouble' else ctypes.c_float
        self.dtype = np.float64 if self.usedtype=='usedouble' else np.float32
        class simulation(ctypes.Structure): #uses the scope of the class by putting it here (Needs, NX,NXP2,ect.)
            """docstring for simulation"""
            _fields_ = [('p1',ctypes.POINTER(c_type*(Nf*NPOP))), #velocity populations 1 of Nf*9
                        ('p2',ctypes.POINTER(c_type*(Nf*NPOP))), #velocity populations 2 of Nf*9 should not be used
                        ('pt1',ctypes.POINTER(c_type*(Nf*NPOPT))), #temperature populations 1 of Nf*4 should not be used
                        ('pt2',ctypes.POINTER(c_type*(Nf*NPOPT))),#temperature populations 1 of Nf*4
                        ('u', ctypes.POINTER(c_type*(Nf))), #x velocity
                        ('v', ctypes.POINTER(c_type*(Nf))), #y velocity
                        ('rho', ctypes.POINTER(c_type*(Nf))), #density
                        ('temperature', ctypes.POINTER(c_type*(Nf))), #temperature
                        ('bottemp', ctypes.POINTER(c_type*(NSIM*NXP2))), #bottom temparature for control
                        ('toptemp', ctypes.POINTER(c_type*(NSIM*NXP2))), #top temparature for control
                        ('t', ctypes.POINTER(ctypes.c_ulong*(NSIM))), #time
                        ('eps',ctypes.POINTER(c_type*(NSIM))), #only use in accelerationcontrol mode
                        ('omega',ctypes.POINTER(c_type*(NSIM))), #only use in accelerationcontrol mode
                        ('gpu', ctypes.c_void_p)] #holds other variables when in cuda mode, cannot be directly accesed

        self.lbmlib = loadlib(self)
        if self.verbose: print('Loading Complete, have a nice day')
        
        #test function
        #self.testfun = wrap_function('testfun', lib=self.lbmlib, restype=ctypes.c_int,
        #                             argtypes=[ctypes.c_int])
        #print('calling test fun...',end='')
        #res = self.testfun(5) #it crashes on print statements
        #print('res=',res)
        self._initfull = wrap_function('initfull', lib=self.lbmlib, restype=ctypes.POINTER(simulation),
                                     argtypes=None)
        if self.verbose: print('starting init')
        self.simpointer = self._initfull()
        if self.verbose: print('initfull done')
        #exit()
        self.p1pointer = self.simpointer.contents.p1
        self.p2pointer = self.simpointer.contents.p2 #should not be used
        self.pt1pointer = self.simpointer.contents.pt1
        self.pt2pointer = self.simpointer.contents.pt2 #should not be used
        self.upointer = self.simpointer.contents.u
        self.vpointer = self.simpointer.contents.v
        self.rhopointer = self.simpointer.contents.rho
        self.temppointer = self.simpointer.contents.temperature
        self.bottemppointer = self.simpointer.contents.bottemp
        self.toptemppointer = self.simpointer.contents.toptemp
        self.tpointer = self.simpointer.contents.t

        
        if self.accelerationcontrol:
            self.epspointer = self.simpointer.contents.eps
            self.omegapointer = self.simpointer.contents.omega
            self.eps = np.ctypeslib.as_array(self.epspointer.contents)
            self.omega = np.ctypeslib.as_array(self.omegapointer.contents)
            self.eps[:] = eps0
            self.omega[:] = omega0
        self.x = np.arange(0.5,self.NX,dtype=self.dtype) #(1/2 to NX-1/2)  domain with ghosts is [.|. . . . .|.] where the walls(|) are zero and NX
        self.y = np.arange(0.5,self.NY,dtype=self.dtype)[::-1] #for plotting, zero = topwall, NY = bottomwall
        self.X,self.Y = np.meshgrid(self.x,self.y) #fixed
        
        self.usource = np.ctypeslib.as_array(self.upointer.contents).reshape((NSIM,NXP2,NYP2))
        self.vsource = np.ctypeslib.as_array(self.vpointer.contents).reshape((NSIM,NXP2,NYP2))
        self.rhosource = np.ctypeslib.as_array(self.rhopointer.contents).reshape((NSIM,NXP2,NYP2))
        self.tempsource = np.ctypeslib.as_array(self.temppointer.contents).reshape((NSIM,NXP2,NYP2))
        if mode=='cpu':
            self.p1source = np.ctypeslib.as_array(self.p1pointer.contents).reshape((NSIM,NXP2,NYP2,NPOP))
            self.pt1source = np.ctypeslib.as_array(self.pt1pointer.contents).reshape((NSIM,NXP2,NYP2,NPOPT))
            self.p2source = np.ctypeslib.as_array(self.p2pointer.contents).reshape((NSIM,NXP2,NYP2,NPOP))
            self.pt2source = np.ctypeslib.as_array(self.pt2pointer.contents).reshape((NSIM,NXP2,NYP2,NPOPT))
        elif mode=='cuda':
            self.p1sourcecuda = np.ctypeslib.as_array(self.p1pointer.contents).reshape((NPOP,NSIM,NXP2,NYP2)) #NPOP,NSIM,NXP2,NYP2 
            self.pt1sourcecuda = np.ctypeslib.as_array(self.pt1pointer.contents).reshape((NPOPT,NSIM,NXP2,NYP2))
            self.p1source = np.moveaxis(self.p1sourcecuda,0,-1) #NPOP,NSIM,NXP2,NYP2 -> NSIM,NXP2,NYP2,NPOP
            self.pt1source = np.moveaxis(self.pt1sourcecuda,0,-1)
        self.bottemp = np.ctypeslib.as_array(self.bottemppointer.contents).reshape((NSIM,NXP2))[:,1:NX+1]
        self.toptemp = np.ctypeslib.as_array(self.toptemppointer.contents).reshape((NSIM,NXP2))[:,1:NX+1]
        self.t = np.ctypeslib.as_array(self.tpointer.contents)
        flipcrop = lambda x: np.swapaxes(np.flip(x[:,1:self.NX+1,1:self.NY+1],axis=2),1,2)
        #remove ghost nodes, flip Y axis swap Y and X axis thus dims become NSIM,NY,NX,...
        # +-----------> x
        # | 0,0 ........ 0,NX-1
        # | .
        # | .
        # v .
        # y NY-1,0 ......NY-1,NX-1
        self.p = flipcrop(self.p1source) 
        self.pt = flipcrop(self.pt1source)
        self.u = flipcrop(self.usource)
        self.v = flipcrop(self.vsource)
        self.rho = flipcrop(self.rhosource)
        self.temp = flipcrop(self.tempsource)
        self._stepsfun = wrap_function('simsteps',lib=self.lbmlib,restype=None,argtypes=[ctypes.POINTER(simulation),
                                                                        ctypes.c_int])
        self._calxyUvrhot = wrap_function('calxyUvrhot',lib=self.lbmlib,restype=None,argtypes=[ctypes.POINTER(simulation)])
        if self.mode=='cuda':
            self._cleanfun = wrap_function('clean',lib=self.lbmlib,restype=None,argtypes=[ctypes.POINTER(simulation)])
            self._copystate = wrap_function('copystate',lib=self.lbmlib,restype=None,argtypes=[ctypes.POINTER(simulation)])
        self.nframes = 0
        self.timing = 0.
        self.warned = False #set by hand to True to supress non-even nsteps even warning
        self.timemem = 0.

        self.window = None
        self.closed = False



    def lonestep(self,simi=0,nsteps=20,bottemp=None,toptemp=None):
        '''Same as allsteps but only for one simulation, can only be used in cpu mode!'''
        assert self.mode == 'cpu', 'only cpu mode for lonestep'
        assert simi<self.NSIM
        if bottemp is not None:
            self.bottemp[simi,:] = bottemp
        if toptemp is not None:
            self.toptemp[simi,:] = toptemp
        lbmlib.lonesteps(self.simpointer,ctypes.c_int(simi),ctypes.c_int(nsteps))
        #self.t[simi] += nsteps*2
        self.updatexyUvrhot()
    
    def allsteps(self,nsteps=40,bottemp=None,toptemp=None,eps=None,omega=None,updatexyUvrhot=True):
        '''Steps all simulations with a fixed given bottemp and toptemp
         nstpes: number of steps, 
            nstpes need to be even.
         bottemp/toptemp: tempature of the boundary
            * None -> temp will remain the same,
            * Number -> temp will be set to that number in all simulations
            * list -> induvisual temperature per simulation
             - if None -> will not change that simulation 
             - if number -> will set temperatur of that simulation
             - if list/array of numbers (length NX) -> will set tempature profile
         updatexyUvrhot: True or False
            if updatexyUvrhot copy the complete state back from gpu to cpu
            Noramlly only rho,t,u,v,pt1,pt3 is copied
            
        returns: time stepping took in seconds
            '''
        if nsteps%2!=0:
            if self.warned==False:
                warnings.warn('\nSetting nsteps={} to an even int will ensure correct behaviour \n \
                    Now will put nsteps into buffer and execute the even number of the buffer.'.format(nsteps))
                self.warned = True
            self.timemem += nsteps
            nsteps = int(self.timemem/2)*2
            self.timemem -= nsteps
            if nsteps==0:
                return 0
        
        #combinations 
        #bottemp is present
        # is none -> list of Nones
        # is number -> list of nums
        # is list (every simulation a different tempature) (could be a number or a tempature profile)
        if eps is not None:
            assert self.accelerationcontrol, 'only when accelerationcontrol=True eps can be set, use lbmd2q9(accelerationcontrol=True)'
            self.eps[:] = eps
        if omega is not None:
            assert self.accelerationcontrol, 'only when accelerationcontrol=True omega can be set, use lbmd2q9(accelerationcontrol=True)'
            self.omega[:] = omega
        if bottemp is None:
            bottemp = [None]*self.NSIM
        elif type(bottemp) is int or type(bottemp) is float:
            bottemp = [float(bottemp)]*self.NSIM #else is a list or numpy array
        if toptemp is None:
            toptemp = [None]*self.NSIM
        elif type(toptemp) is int or type(toptemp) is float:
            toptemp = [float(toptemp)]*self.NSIM
        for simi,tbot,ttop in zip(list(range(self.NSIM)),bottemp,toptemp):
            if tbot is not None:
                self.bottemp[simi,:] = tbot 
            if ttop is not None:
                self.toptemp[simi,:] = ttop
        starttime = time.time()
        self._stepsfun(self.simpointer,ctypes.c_int(int(nsteps/2))) #will by defalth also copy rho,temp,u,v,pt1,pt3 back to cpu if in cuda
        self.timing = time.time()-starttime
        if self.mode=='cuda' and updatexyUvrhot: #will also copy complete state
            self.updatexyUvrhot()

        return self.timing

    def updatexyUvrhot(self):
        '''updated u,v,rho,tempature and pull populations from device if on cuda'''
        self._calxyUvrhot(self.simpointer)

    def clean(self):
        assert self.closed is False, 'already closed'
        '''Good practice to use clean when using cuda'''
        if self.verbose: print('cleaning...')
        if self.mode=='cuda':
            self._cleanfun(self.simpointer)
        self.closerender()
        self.closed = True

    def close(self):
        '''Same as clean'''
        self.clean()
    
    ############## plotting ################
    def flowplot(self,simi=0,simulationname=None,vmin=0,vmax=0.02):
        plt.figure(figsize=(15,10))
        plt.quiver(self.u[simi],self.v[simi])
        plt.imshow(np.sqrt(self.u[simi]**2+self.v[simi]**2),vmin=vmin,vmax=vmax)
        plt.colorbar()
        if simulationname:
            plt.savefig(simulationname+'figflow{:09d}'.format(self.t[simi])+'.jpg')
            plt.close()
        else:
            plt.show()
    def tempplot(self,simi=0,simulationname=None,vmin=None,vmax=None,figsize=(16,9)):
        plt.figure(figsize=figsize)
        plt.imshow(self.temp[simi],vmin=vmin,vmax=vmax)
        plt.colorbar()
        if simulationname:
            plt.savefig(simulationname+'figtemp{:09d}.jpg'.format(self.t[simi]))
            plt.close()
        else:
            plt.show()
    def flowtemplot(self,simi=0,simulationname=None,vmin=None,vmax=None):
        plt.figure(figsize=(15,10))
        plt.quiver(self.u[simi],self.v[simi])
        plt.imshow(self.temp[simi],vmin=vmin,vmax=vmax)
        plt.colorbar()
        if simulationname:
            plt.savefig(simulationname+'quiverT{:09d}'.format(self.t[simi])+'.jpg')
            plt.close()
        else:
            plt.show()

    #####################  animations  ###########################
    def addframe(self,title='',simi=0,figsize=(8,8),namesave=None):
        '''usege example: (also see controlRB.test_animation() function)
        lbm = lbmd2q9(NX=200,NY=32,NSIM=1)
        for i in tqdm(range(400)):
            lbm.allsteps(nsteps=40)
            lbm.addframe(figsize=(15,3))
        lbm.tovid()'''
        global fig
        if self.nframes==0:
            #remove frames from 
            fig = plt.figure(figsize=figsize)
            self.ims = []
        
        plt.title(title)
        plt.set_cmap('hot')
        im = plt.imshow(np.copy(self.temp[simi]), animated=True,vmin=1,vmax=2)
        if self.nframes==0: 
            plt.axis('off')
            plt.tight_layout()
        self.ims.append([im])
        if namesave: fig.savefig(namesave + '{:03d}.jpg'.format(self.nframes))
        self.nframes+=1
    def tovid(self,save=False,name="movie.mp4",interval=40):
        '''usege example: (also see test_animation() function)
        lbm = lbmd2q9(NX=200,NY=32,NSIM=1)
        for i in tqdm(range(400)):
            lbm.allsteps(nsteps=40)
            lbm.addframe(figsize=(15,3))
        lbm.tovid()'''
        global fig
        if self.verbose: print('making video...')
        ani = animation.ArtistAnimation(fig, self.ims, interval=interval, blit=True,
                                repeat_delay=1000)
        if save:
            if self.verbose: print('saving video... as {}'.format(name))
            #mywriter = animation.FFMpegWriter()
            ani.save(name)
            plt.close()
            self.nframes = 0
        else:
            if self.verbose: print('converting to HTML video')
            assert HTMLloaded, 'HTML could not be imported but needed to be loaded to save video to HTML'
            vid = HTML(ani.to_html5_video())
            plt.close()
            self.nframes = 0
            return vid

    #############Saving/loading##################
    def savestateall(self,name):
        self.updatexyUvrhot()
        if (name[-4:])!='.npz':
            name = name + '.npz'
        np.savez(name,p1source=self.p1source,
                 pt1source=self.pt1source,
                 usource=self.usource,
                 vsource=self.vsource,
                 rhosource=self.rhosource,
                 tempsource=self.tempsource,
                 t=self.t)
    def loadstateall(self,name):
        if (name[-4:])!='.npz':
            name = name + '.npz'
        out = np.load(name)
        self.p1source[:,:,:,:] = out['p1source']
        self.pt1source[:,:,:,:] = out['pt1source']
        self.usource[:,:,:] = out['usource']
        self.vsource[:,:,:] = out['vsource']
        self.rhosource[:,:,:] = out['rhosource']
        self.tempsource[:,:,:] = out['tempsource']
        self.t[:] = out['t']
        if self.mode=='cuda':
            self._copystate(self.simpointer)
    def savestatelone(self,name,simi=0):
        self.updatexyUvrhot()
        if (name[-4:])!='.npz':
            name += '.npz'
        np.savez(name,p1source=self.p1source[simi],
                 pt1source=self.pt1source[simi],
                 usource=self.usource[simi],
                 vsource=self.vsource[simi],
                 rhosource=self.rhosource[simi],
                 tempsource=self.tempsource[simi],
                 t=self.t[simi])
    def loadstatelone(self,name,simi=0):
        if self.mode=='cuda':
            self.updatexyUvrhot()
        if (name[-4:])!='.npz':
            name += '.npz'
        out = np.load(name)
        self.p1source[simi,:,:,:] = out['p1source']
        self.pt1source[simi,:,:,:] = out['pt1source']
        self.usource[simi,:,:] = out['usource']
        self.vsource[simi,:,:] = out['vsource']
        self.rhosource[simi,:,:] = out['rhosource']
        self.tempsource[simi,:,:] = out['tempsource']
        self.t[simi] = int(out['t'])
        if self.mode=='cuda':
            if self.verbose: print('maybe consider implement a function for this that does not copy simulation state?')
            self._copystate(self.simpointer)
        
    ############# UTILS ####################
    def calNu(self,simi=0,h=None): ## h=None nusselts
        """calculates Nusselts on height h, if h is None it is an average, calNuall is a lot faster!"""
        tbot = np.mean(self.bottemp[simi,:])
        ttop = np.mean(self.toptemp[simi,:])
        Qcon = self.kappa*(tbot-ttop)/(self.NY)
        if h is None:
            if self.mode=='cpu' or self.mode=='cuda':
                Nu = np.mean([np.mean(self.pt1source[simi,1:-1,h,1]-self.pt1source[simi,1:-1,h+1,3])/Qcon \
                              for h in range(0,self.NY+1)])
        else:
            if self.mode=='cpu' or self.mode=='cuda':
                Nu = np.mean(self.pt1source[simi,1:-1,h,1]-self.pt1source[simi,1:-1,h+1,3])/Qcon
        return Nu

    def calNuall(self):
        """Calculates the averaged nusselts number for all simulations, is quite a bit faster than calNu"""
        tbot = np.mean(self.bottemp,axis=1)
        ttop = np.mean(self.toptemp,axis=1)
        Qcon = self.kappa*(tbot-ttop)/(self.NY)
        #Qcon = self.kappa*(self.TBOT0-self.TTOP0)/(self.NY)
        return np.mean(self.pt1source[:,1:-1,0:-1,1]-self.pt1source[:,1:-1,1:,3],axis=(1,2))/Qcon

    ########### pyglet visualisation ########
    def render(self,windowsize=600,vmin=None,vmax=None):
        '''Faster visualisation than just the plots using pyglet
         NX > NY'''

        import pyglet
        if self.window is None:
            self.window = pyglet.window.Window(width=windowsize, height=windowsize,vsync=False,resizable=True)
            self.windowwidth = windowsize
            self.windowheight = windowsize



        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        #rescale
        import skimage.transform
        # make center image from current state named ar
        arr = skimage.transform.rescale(self.temp[0],float(self.windowwidth)/self.temp.shape[1],
                                        anti_aliasing=False,mode='reflect',multichannel=False) #rescaled
        Max = np.max(arr) if vmax is None else vmax
        Min = np.min(arr) if vmin is None else vmin
        arr = np.array(np.clip((arr-Min)/((Max-Min)+1e-15)*255,0,255),dtype=np.uint8)
        arr = arr[:,:,np.newaxis]*np.array([1,1,1],np.uint8) #new axis with 3
        NYimg = arr.shape[0]

        # Convert array to texture
        img = pyglet.image.ImageData(arr.shape[1],arr.shape[0],'RGB',arr.tobytes(),pitch=arr.shape[1]*-3) #potential error in arr.shape
        texture = img.get_texture()
        

        #window.flip()
        a = float(self.windowwidth)/2-NYimg/2
        texture.blit(0, a)
        self.window.flip()

    def closerender(self):
        if self.window is not None:
            self.window.close()
            self.window = None

if __name__ == '__main__':
    lbm = lbmd2q9()
