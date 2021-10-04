from controlRB.LBMd2q9ctypes import lbmd2q9, plt, np
from controlRB.misc import tqdm


########### Simulation Validation ##############
def validate_Poiseuille(size=32,mode='cpu',file=None,usedtype='double',returndata=False):
    '''Validate Simulation using comparason with Poiseuille analitical solution'''
    if file:
        file = 'Poiseuille.jpg'
    lbm = lbmd2q9(mode=mode,NX=size,NY=size,NSIM=1,TBOT0=1.,TTOP0=1.,
                  wallprop='PERIODIC',Ra=0.,ff_body=0.00001,verbose=True,usedtype=usedtype)
    """Does nsteps steps than compares the result the reference solution"""
    nsteps = int(4000*4*size/32)*4
    lbm.allsteps(nsteps=nsteps)
    lbm.U0 = lbm.ff_body*lbm.NY**2/(lbm.nu*8)
    ysimu = np.linspace(0.5,lbm.NY-0.5,num=lbm.NY)
    yref = np.linspace(0,lbm.NY,num=200)
    Uref = lbm.U0*(1-((yref-lbm.NY/2)/(lbm.NY/2))**2)
    Usimi = lbm.u[0,:,int(lbm.NX/2)]
    if returndata:
        return dict(Uref=Uref,yref=yref,Usimi=Usimi,ysimu=ysimu,diff=Usimi-lbm.U0*(1-((ysimu-lbm.NY/2)/(lbm.NY/2))**2))
    plt.plot(Uref,yref)
    plt.plot(Usimi,ysimu,'.')
    plt.ylabel('height')
    plt.xlabel('flow u')
    plt.title('Poiseuille flow')
    plt.legend(['Theory','Simulation'])
    if file:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()
    diff = Usimi-lbm.U0*(1-((ysimu-lbm.NY/2)/(lbm.NY/2))**2)
    error = np.max(diff**2)**0.5
    print('PoiseuilleVali error =', error,'=',error/5.0040621623954895e-05*100-100,'%')
    print('expected = 5.0040621623954895e-05')
    return lbm


def validate_thermaldiff(file=False,mode='cpu',NY=64,Pr=0.71,usedtype='double',returndata=False):
    '''Validate simulation using thermal diffusion. '''
    def analytic(lbm,ylist,t):
        data = []
        for y in ylist:
            tot = lbm.TBOT0+(lbm.TTOP0-lbm.TBOT0)/lbm.NY*y
            for m in range(2,200,2):
                tot += np.sin(m*np.pi*y/lbm.NY)*(lbm.TTOP0-lbm.TBOT0)/ \
                (m/2*np.pi)*np.exp(-(np.pi*m/lbm.NY)**2*lbm.kappa*t)
            data.append(tot)
        return data
    lbm = lbmd2q9(Ra=0,NX=8,NY=NY,Pr=Pr,mode=mode,usedtype=usedtype)
    dt = int(500*(NY/64)**2)*2
    i=0
    ysimu = np.arange(0.5,lbm.NY)
    tanalitics = []
    tsimus = []
    for t in range(dt,dt*5,dt):
        i+=1
        lbm.allsteps(nsteps=dt)
        tanalitic = analytic(lbm,ysimu,t)
        plt.plot(ysimu,tanalitic)
        tanalitics.append(tanalitic)
        tsimu = np.flip(np.mean(lbm.temp[0],axis=1))
        error = np.max((tanalitic-tsimu)**2)**0.5
        print('at t=',t,' error=',error)
        print('expected = 0.004','thus this is',error/0.0038563761979601985*100-100,'%')
        plt.plot(ysimu,tsimu)
        tsimus.append(tsimu)
        plt.title('t={}'.format(t))
        plt.legend(['analytical','simulation'])
        plt.xlabel('height')
        plt.ylabel('temperature')
        if file:
            plt.savefig('difffig{}.jpg'.format(i))
            plt.close()
        else:
            plt.show()
    if returndata:
        return dict(ysimu=ysimu,tanalitics=np.array(tanalitics),tsimus=np.array(tsimus))
    return lbm

def validate_Ra1e5(size=64,mode='cpu',file=False,usedtype='double',returndata=False):
    '''Validate simulation using the reference Nuseltz 1e5 number of 
    Numerical simulation of two-dimensional Rayleigh–Bénard convection in an enclosure
    Nasreddine Ouertatani, Nader Ben Cheikh Brahim Ben Beya, Taieb Lili, 2008'''
    lbm = lbmd2q9(mode=mode,NX=size,NY=size,NSIM=1,TBOT0=2.,TTOP0=1.,
                  wallprop='WALLED',Ra=1e5,ff_body=0.0,verbose=True,usedtype=usedtype)
    lbm.allsteps(nsteps=int(10**4*size/64)*2)
    dataNus = []
    for i in range(50):
        lbm.allsteps(nsteps=int(500*size/64)*2)
        dataNus.append(lbm.calNu())
    Nuh = [lbm.calNu(h=hi) for hi in range(1,lbm.NY-1)]
    if returndata:
        return dict(dataNus=np.array(dataNus),Nuh=np.array(Nuh))
    plt.plot(dataNus)
    plt.xlabel('time')
    if file:
        plt.savefig('Ra1e5time.jpg')
        plt.close()
    else:
        plt.show()
    print('Nu=',dataNus[-1],'from ref Nu=',3.91)
    plt.plot(Nuh)
    plt.xlabel('height (y)')
    plt.ylabel('Nu')
    if file:
        plt.savefig('Ra1e5Height.jpg')
        plt.close()
    else:
        plt.show()
    return lbm

def validate_transision(size=32,plotwhile=True,usedtype='double',returndata=False):
    '''Validate simulation using the reference Nuseltz number of 
    Numerical simulation of two-dimensional Rayleigh–Bénard convection in an enclosure
    Nasreddine Ouertatani, Nader Ben Cheikh Brahim Ben Beya, Taieb Lili, 2008'''
    Ralist = []
    Nulist = []
    Nustdlist = []
    for Ra in tqdm(np.logspace(1,6,num=16)):
        lbm = lbmd2q9(NX=size,NY=size,Ra=Ra,verbose=False,Pr=0.71,usedtype=usedtype) #2e6 for unstable 
        lbm.allsteps(nsteps=int(10000*size/32)*2)
        data = []
        for i in range(10):
            lbm.allsteps(nsteps=int(2000*size/32)*2)
            data.append(lbm.calNu())
        Nulist.append(max(1,np.mean(data)))
        Nustdlist.append(np.std(data))
        Ralist.append(Ra)
        if len(Ralist)>1 and plotwhile:
            plt.semilogx(Ralist,Nulist,'o--')
            plt.semilogx(Ralist,np.array(Nulist)+np.array(Nustdlist),'.')
            plt.semilogx(Ralist,np.array(Nulist)-np.array(Nustdlist),'.')
            plt.xlabel('Ra')
            plt.ylabel('Nu')
            plt.show()
    plt.figure(figsize=(8,6))
    Raref = np.logspace(1,6,num=100)
    a,b = np.polyfit(np.log(Ralist[7:-1]),Nulist[7:-1],1)
    Nuref =  np.log(Raref)*a+b
    plt.semilogx(Ralist,Nulist,'.')
    Rarefs = [1e3,1e4,1e5,1e6]
    Nurefs = [1.0004,2.1580,3.9103,6.3092]
    plt.semilogx(Rarefs,Nurefs,'.') #use of reference
    plt.semilogx(Raref,Nuref)
    plt.legend(['Nu simu','Nu ref','extrapolation fit'])
    plt.xlabel('Ra')
    plt.ylabel('Nu')
    lims = list(plt.axis())
    lims[2] = -0.2
    plt.axis(lims)
    plt.show()
    Ralistforref = Ralist[6],Ralist[9],Ralist[12],Ralist[15]
    Nucal = Nulist[6],Nulist[9],Nulist[12],Nulist[15]
    Nureff = [1.0004,2.15,3.9103,6.3]
    for Ranow,Nuc,Nur in zip(Ralistforref,Nucal,Nureff):
        print(Ranow,Nuc,Nur)
    
    return dict(Ralist=Ralist,Nulist=Nulist,Nustdlist=Nustdlist,Rarefs=Rarefs,Nurefs=Nurefs)



######   control BC #####
def test_temps(usedtype='double'):
    """test controlable BC and show that they are working correctly
    This is that both simulation that get started can be controlled independently"""
    lbm = lbmd2q9(NSIM=2,usedtype=usedtype)
    lbm.allsteps()
    print('simi 0: Vanilla')
    lbm.flowtemplot()
    print('simi 1: Vanilla')
    lbm.flowtemplot(simi=1)
    lbm.allsteps(bottemp=1,toptemp=1)
    print('simi 0: top=bot=1')
    lbm.flowtemplot()
    print('simi 1: top=bot=1')
    lbm.flowtemplot(simi=1)
    lbm.allsteps(nsteps=400,bottemp=[1,2],toptemp=[2,1])
    print('simi 0: bot=1, top=2')
    lbm.flowtemplot()
    print('simi 1: bot=2, top=1')
    lbm.flowtemplot(simi=1)
    lbm.allsteps(nsteps=400,bottemp=[2,1+np.sin(np.arange(0,2*np.pi,2*np.pi/lbm.NX))],
                 toptemp=[1,1-np.sin(np.arange(0,2*np.pi,2*np.pi/lbm.NX))])
    print('simi 0: bot=2 top=1')
    lbm.flowtemplot()
    print('simi 1: bot=1+sin, top = 1-sin')
    lbm.flowtemplot(simi=1)

def test_animation(usedtype='double'):
    '''Test the animation module and return the video as a html vid, best viewed in jupyter notebooks'''
    lbm = lbmd2q9(NX=200,NY=32,NSIM=1,usedtype=usedtype)
    for i in tqdm(range(200)):
        lbm.allsteps(nsteps=40)
        lbm.addframe(figsize=(15,3))
    vid = lbm.tovid()
    return vid