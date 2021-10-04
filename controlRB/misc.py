

from controlRB.LBMd2q9ctypes import lbmd2q9

import numpy as np
import time #timing of functions

try:
    import warnings
    import tqdm as tqdmmod
    warnings.filterwarnings("ignore",category=tqdmmod.TqdmExperimentalWarning) #suppres warning
    try:
        from tqdm.autonotebook import tqdm #works for both jupyter notebooks and consols
    except: #autonotebook is experimental
        from tqdm import tqdm
except ImportError:
    print('failed to load tqdm')

try:
    from matplotlib import pyplot as plt
    try:
        plt.plot([0,1]) #this will create RuntimeError error on volta
    except RuntimeError:  #plt.switch_backend('agg') #need to be active for volta plotting
        plt.switch_backend('agg') #this does not work on my laptop thus I made this weird construct
                                  #better solution would be preferred.
    plt.close()
    import matplotlib.cbook
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    import matplotlib.animation as animation #cannot be used on volta
except ImportError:
    print('failed to load matplotlib')
try:
    from IPython.display import HTML #HTML videos
    HTMLloaded = True
except ImportError:
    HTMLloaded = False

# timescales to get to steady state in NX=64 and NY=64 for different Ra, 
# can be used in preconditioning the state
timescalesref = np.array([[1e3,15e3],[2e3,15e3],[2.8e3,1.5e6],[3e3,8e5],[4e3,2e5],[5e3,2e5],[7e3,1e5],[1e4,60e3],[2e4,40e3],[5e4,50e3],
                          [1e5,30e3],[2e5,30e3],[5e5,30e3], [1e6,25e3],[5e6,20e3],[1e7,40e3],[3e7,80e3]])
# minimal sizes
# 1e4 -> 20
# 1e5 -> 30
# 1e6 -> 64
# 3e6 -> 200
# 1e7 -> 300

from scipy import interpolate
timescaleRafun = interpolate.interp1d(timescalesref[:,0],timescalesref[:,1]) #scale function
#timescaleRafunwithNX = lambda Ra,NX: timescaleRafun(Ra)*(NX/64) #is this correct?

#never really used but the mean kinetic energy for a NX=NY=64 system for different Ra
kinscaleref = np.array([[1000.0, 1.634522009039148e-13],
 [3162.2776601683795, 3.780002365782981e-12], #this is not correct
 [10000.0, 5.354127894938509e-05],
 [31622.776601683792, 0.00024878479754503836],
 [100000.0, 0.0009474146756246959],
 [316227.7660168379, 0.003241915111589941],
 [1000000.0, 0.010494684906559668]])
kinscalefun = interpolate.interp1d(kinscaleref[:,0],kinscaleref[:,1])


def simuinteract(NX=64,NY=64,Ra=1e5,wallprop='WALLED',modependulum=False,modeaccerlation=False,modemeanvar=False,verbose=False,figsize=(8,8)):
    '''function to look at the classical control using ipywidgets
       Interact with simulation only avaiable when in jupyter notebook mode
       note: this will make a startpos.npz file in /tmp/'''
    assert modependulum or modeaccerlation or modemeanvar, 'please choose a mode'
    assert int(modependulum)+int(modeaccerlation)+int(modemeanvar)==1, 'please choose only one mode'
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    if modependulum or modemeanvar:
        lbm = lbmd2q9(NX=NX,NY=NY,Ra=Ra,wallprop=wallprop,verbose=verbose)
    elif modeaccerlation:
        lbm = lbmd2q9(NX=NX,NY=NY,Ra=Ra,wallprop=wallprop,accelerationcontrol=True,verbose=verbose)
    lbm.savestatelone('/tmp/startpos')

    def steppendulum(name='',reset=False,nsteps=40,Amplitude=0.,Period=2000,tempplot=False,flowtemplot=False,
             contourplot=False,tempmovie=False,ntempframes=100):
        A = Amplitude
        P = Period
        if reset: lbm.loadstatelone('/tmp/startpos')
        ts = np.array(np.round(np.linspace(0,nsteps,num=int(np.ceil(nsteps/(P/10)+1)))/2)*2,dtype=np.int)
        dts = ts[1:]-ts[:-1]
        calbottemp = lambda t: [[lbm.TBOT0+A*np.sin(np.pi*2/P*t)]*(int(lbm.NX/2)) + \
                                [lbm.TBOT0-A*np.sin(np.pi*2/P*t)]*(int(lbm.NX/2))]
        if tempmovie:
            for i in tqdm(range(ntempframes)):
                for dt in dts:
                    t = lbm.t[0]
                    t += dt
                    lbm.allsteps(nsteps=dt,bottemp=calbottemp(t))
                lbm.addframe(title=name + " Ra={:.2E},A={:.2f},P={:.2E}".format(lbm.Ra,A,P),figsize=figsize)
            return lbm.tovid()
        else:
            for dt in dts:
                t = lbm.t[0]
                t += dt
                lbm.allsteps(nsteps=dt,bottemp=calbottemp(t))
            if contourplot:
                plt.figure(figsize=(8,6))
                CS = plt.contour(lbm.temp[0,::-1,:],np.linspace(1,2,num=16))
                plt.clabel(CS)
                plt.show()
            if flowtemplot:
                lbm.flowtemplot(vmin=1.,vmax=2.)
            if tempplot:
                lbm.tempplot(vmin=1.,vmax=2.)

    def stepaccerlation(name='',reset=False,nsteps=40,eps=0.,Period=2000,tempplot=False,flowtemplot=False,
             contourplot=False,tempmovie=False,ntempframes=100,doubleplot=False):
        if reset: lbm.loadstatelone('/tmp/startpos')
        lbm.omega[:] = np.pi*2/Period
        lbm.eps[:] = eps
        if tempmovie:
            for i in tqdm(range(ntempframes)):
                lbm.allsteps(nsteps=nsteps)
                lbm.addframe(title=name + " Ra={:.2E},eps={:.2f},P={:.2E}".format(lbm.Ra,eps,Period),doubleplot=doubleplot,figsize=figsize)
            return lbm.tovid()
        else:
            lbm.allsteps(nsteps=nsteps)
            if contourplot:
                plt.figure(figsize=(8,6))
                CS = plt.contour(lbm.temp[0,::-1,:],np.linspace(1,2,num=16))
                plt.clabel(CS)
                plt.show()
            if flowtemplot:
                lbm.flowtemplot(vmin=1.,vmax=2.)
            if tempplot:
                lbm.tempplot(vmin=1.,vmax=2.)

    def stepmeanvar(name='',reset=False,nsteps=40,eps=0.,Period=2000,tempplot=False,flowtemplot=False,
             contourplot=False,tempmovie=False,ntempframes=100,single=True):
        P = Period
        if reset: lbm.loadstatelone('/tmp/startpos')
        omega = np.pi*2/P
        ts = np.array(np.round(np.linspace(0,nsteps,num=int(np.ceil(nsteps/(P/10)+1)))/2)*2,dtype=np.int)
        dts = ts[1:]-ts[:-1]
        if single:
            calbottemp = lambda lbm,dt: float(2+eps*np.sin((lbm.t[0]+dt/2)*omega))
            caltoptemp = lambda lbm,dt: float(1)
        else:
            calbottemp = lambda lbm,dt: float(2+eps/2*np.sin((lbm.t[0]+dt/2)*omega))
            caltoptemp = lambda lbm,dt: float(1-eps/2*np.sin((lbm.t[0]+dt/2)*omega))
        if tempmovie:
            for i in tqdm(range(ntempframes)):
                for dt in dts:
                    t = lbm.t[0]
                    t += dt
                    lbm.allsteps(nsteps=dt,bottemp=calbottemp(lbm,dt),toptemp=caltoptemp(lbm,dt))
                lbm.addframe(title=name + " Ra={:.2E},eps={:.2f},P={:.2E}".format(lbm.Ra,eps,Period),figsize=figsize)
            return lbm.tovid()
        else:
            for dt in dts:
                t = lbm.t[0]
                t += dt
                lbm.allsteps(nsteps=dt,bottemp=calbottemp(lbm,dt),toptemp=caltoptemp(lbm,dt))
            if contourplot:
                plt.figure(figsize=(8,6))
                CS = plt.contour(lbm.temp[0,::-1,:],np.linspace(1,2,num=16))
                plt.clabel(CS)
                plt.show()
            if flowtemplot:
                lbm.flowtemplot(vmin=1.,vmax=2.)
            if tempplot:
                lbm.tempplot(vmin=1.,vmax=2.)

    if modependulum:
        interact_manual(steppendulum,nsteps=(20,50000,4),Amplitude=(0.,1.,0.00001),Period=(400,50000,20),ntempframes=(10,300));
    elif modeaccerlation:
        interact_manual(stepaccerlation,nsteps=(20,50000,4),eps=(0.,10.,0.1),Period=(400,50000,20),ntempframes=(10,300));
    elif modemeanvar:
        interact_manual(stepmeanvar,nsteps=(20,50000,4),eps=(0.,10.,0.1),Period=(400,50000,20),ntempframes=(10,300));
    else:
        assert False, 'wat?'

def plotmaker(Ra,eps,omega,modeaccerlation=False,modemeanvarsingle=False,modemeanvardouble=False,lbm=None):
    '''For a given classical pasive control method it will it will generate 
    a 1: video, 
      2: Nu(y), Kin(y), T(y), 
      3: Nu(t), Kin(t), T(t)'''
    assert int(modeaccerlation)+int(modemeanvarsingle)+int(modemeanvardouble)==1, 'one mode should be selected'
    
    # units
    if lbm==None:
        print('lbm not present setting up the simulation')
        lbm = lbmd2q9(Ra=Ra,accelerationcontrol=modeaccerlation) #set eps,omega later
        lbmpresent = False
    else:
        print('lbm present')
        lbmpresent = True
    #omega = omegadim*lbm.NY**2/lbm.kappa
    omegadim = omega*lbm.kappa/lbm.NY**2
    Period = int(np.pi*2/omegadim/2)*2 #is even
    print('Period=',Period)


    if modeaccerlation:
        lbm.eps[:] = eps
        lbm.omega[:] = omegadim

    if modeaccerlation:
        caltoptemp = lambda t: lbm.TTOP0
        calbottemp = lambda t: lbm.TBOT0
    elif modemeanvarsingle:
        caltoptemp = lambda t: float(lbm.TTOP0)
        calbottemp = lambda t: float(lbm.TBOT0 + eps*np.sin(omegadim*t))#or cos?
    elif modemeanvardouble:
        caltoptemp = lambda t: float(lbm.TTOP0 - eps/2*np.sin(omegadim*t))
        calbottemp = lambda t: float(lbm.TBOT0 + eps/2*np.sin(omegadim*t))
    # startpresteps

    if lbmpresent==False: #setup the state if lbm is not present
        pretimeteps = max(np.ceil(timescaleRafun(Ra)*2/Period)*Period,Period*5)
        print('pretimeteps=',pretimeteps)
        #get dts
        dtmax = int(Period/15/2)*2
        dtlist = np.linspace(0,pretimeteps/2,num=int(pretimeteps/dtmax)+2,dtype=np.int)*2
        dtlist = dtlist[1:]-dtlist[:-1]
        for dt in tqdm(dtlist):
            lbm.allsteps(nsteps=dt,bottemp=calbottemp(lbm.t[0]+dt/2),toptemp=caltoptemp(lbm.t[0]+dt/2),updatexyUvrhot=False)
    else:
        print('no presteps')

    # make video
    pretimeteps = Period*2
    #get new dts

    dtmax = int(Period/50/2)*2
    dtlist = np.linspace(0,pretimeteps/2,num=int(pretimeteps/dtmax)+2,dtype=np.int)*2
    dtlist = dtlist[1:]-dtlist[:-1]
    for dt in tqdm(dtlist):
        lbm.allsteps(nsteps=dt,bottemp=calbottemp(lbm.t[0]+dt/2),toptemp=caltoptemp(lbm.t[0]+dt/2),updatexyUvrhot=True)
        lbm.addframe()
    print("making video")
    vid = lbm.tovid()


    # Space/time averaging averaging  Nu(y), Kin(y), T(y), 3: Nu(t), Kin(t), T(t) -> Nu(t,y),Kin(t,y),T(t,y)
    print('starting space averaging')
    pretimeteps = Period*2
    dtmax = int(Period/100/2)*2
    dtlist = np.linspace(0,pretimeteps/2,num=int(pretimeteps/dtmax)+2,dtype=np.int)*2
    dtlist = dtlist[1:]-dtlist[:-1]

    Nulist = []
    Kinlist = []
    Tlist = []
    epsulist = []
    for dt in tqdm(dtlist):
        lbm.allsteps(nsteps=dt,bottemp=calbottemp(lbm.t[0]+dt/2),toptemp=caltoptemp(lbm.t[0]+dt/2),updatexyUvrhot=True)
        Nulist.append([lbm.calNu(h=h) for h in range(0,lbm.NY-1)])
        Kin = 0.5*lbm.rho[0]*(lbm.u[0]**2+lbm.v[0]**2)
        epsu1 = lbm.nu*((lbm.u[0,:,1:]-lbm.u[0,:,:-1])**2 + \
                (lbm.v[0,:,1:]-lbm.v[0,:,:-1])**2)
        epsu2 = lbm.nu*((lbm.u[0,1:,:]-lbm.u[0,:-1,:])**2 + \
                (lbm.v[0,1:,:]-lbm.v[0,:-1,:])**2)
        epsulist.append(np.mean([np.mean(epsu1),np.mean(epsu2)]))
        Kinlist.append(np.mean(Kin,axis=1))
        Tlist.append(np.mean(lbm.temp[0],axis=1))
    Nulist = np.array(Nulist)
    Kinlist = np.array(Kinlist)
    Tlist = np.array(Tlist)
    epsulist = np.array(epsulist)

    Nut = np.mean(Nulist,axis=1)
    Kint = np.mean(Kinlist,axis=1)
    Tt = np.mean(Tlist,axis=1)
    epsut = epsulist
    Nu = np.mean(Nut)
    Pr = lbm.Pr
    nu = lbm.nu
    L = lbm.NY
    print('\n\n Mean Quantaties')
    print('Ra = ',lbm.Ra)
    print('Pr = ',lbm.Pr)
    print('Nu = ',Nu)
    print('Kin = ',np.mean(Kint))
    print('T = ',np.mean(Tt))
    print('epsu = ',np.mean(epsut))
    print('(Nu-1)*Ra*Pr^-2 nu^3/L^4 = ',(Nu-1)*Ra*Pr**-2*nu**3/L**4)

    Nuy = np.mean(Nulist,axis=0)
    Kiny = np.mean(Kinlist,axis=0)
    Ty = np.mean(Tlist,axis=0)

    print('time 2 periods:')
    plt.plot(Nut)
    plt.title('Nut')
    plt.show()
    plt.plot(Kint)
    plt.title('Kint')
    plt.show()
    plt.plot(Tt)
    plt.title('Tt')
    plt.show()
    plt.title('epsu')
    plt.plot(epsut)
    plt.show()

    print('Height:')
    plt.plot(Nuy)
    plt.title('Nuy')
    plt.show()
    plt.plot(Kiny)
    plt.title('Kiny')
    plt.show()
    plt.plot(Ty)
    plt.title('Ty')
    plt.show()
    return lbm,vid,Nut,Kint,Tt,Nuy,Kiny,Ty