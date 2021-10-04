// this files hold the displace, collide, bc kernals

void displace(pop * p1, pop * p2, popt * pt1, popt * pt2) { // streams p1 to p2 and pt1 to pt2
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
  int x, y, idx;
  int xm, xp, ym, yp;

  for (y=1; y<NY+1; y++) {
    ym = y-1;
    yp = y+1;
    for (x=1; x<NX+1; x++) {
      xm = x-1;
      xp = x+1;
      idx = IDX(y, x);

      p2[idx].p[0] = p1[IDX(y,   x)].p[0]; 
      p2[idx].p[1] = p1[IDX(y,  xm)].p[1]; 
      p2[idx].p[5] = p1[IDX(ym, xm)].p[5]; 
      p2[idx].p[2] = p1[IDX(ym,  x)].p[2]; 
      p2[idx].p[6] = p1[IDX(ym, xp)].p[6]; 
      p2[idx].p[3] = p1[IDX(y,  xp)].p[3]; 
      p2[idx].p[7] = p1[IDX(yp, xp)].p[7]; 
      p2[idx].p[4] = p1[IDX(yp,  x)].p[4]; 
      p2[idx].p[8] = p1[IDX(yp, xm)].p[8]; 

      pt2[idx].p[0] = pt1[IDX(y, xm)].p[0]; 
      pt2[idx].p[1] = pt1[IDX(ym,  x)].p[1]; 
      pt2[idx].p[2] = pt1[IDX(y,  xp)].p[2]; 
      pt2[idx].p[3] = pt1[IDX(yp,  x)].p[3]; 
    }  
  }
}

#ifdef ACCELERATION
void collide(pop * p, popt * pt, data_t epssin) // and force
#else
void collide(pop * p, popt * pt) // and force
#endif
{
   int x, y, pp, idx;
   data_t u, v, T;
   data_t u2,v2, u2d2cs4, v2d2cs4;
   data_t sumsq, sumsq2;
   data_t invtau, rho;
   data_t ui, vi, uv;
   pop p_eq, R;
   popt pt_eq;
   data_t invtaut,ff_buoy,tauRdrho,ff_bodyrho,Rterm;

   invtau = 1.0 / tau;
   invtaut = 1.0 / taut;
   // d2q4: Numerical study of lattice Boltzmann methods for a convection–diffusion equation coupled with Navier–Stokes equations
   // forcing: book: The Lattice Boltzmann Method Principles and Practice in  
   //          6.4 Alternative Forcing Schemes (p. 241) method: He et al. [39] 
   for (y=1; y<NY+1; y++) { // on the bulk
      for (x=1; x<NX+1; x++) {
         idx = IDX(y, x);
         T = vt(pt[idx]);
         rho = m(p[idx]);
         // force = He et al. [39]
#ifdef ACCELERATION
         ff_buoy = rho*alphag*(T-T0)*(1.+epssin); // added acceraltion term
#else
         ff_buoy = rho*alphag*(T-T0);
#endif
         ff_bodyrho = ff_body*rho;

         u = (vx(p[idx]) + 0.5*ff_bodyrho) / rho;
         v = (vy(p[idx]) + 0.5*ff_buoy) / rho;
         u2 = u * u;
         v2 = v * v;

         sumsq  = (u2 + v2) / cs22;
         sumsq2 = sumsq * (1.0 - cs2) * cs2i;
         u2d2cs4 = u2 / cssq; // cssq = 2/9 = 2*cs2**2
         v2d2cs4 = v2 / cssq;

         ui = u * cs2i;
         vi = v * cs2i;
         uv = ui * vi;

         p_eq.p[0] = rho * rt0 * (1.0 - sumsq); //correct
         p_eq.p[1] = rho * rt1 * (1.0 - sumsq + u2d2cs4 + ui); //correct
         p_eq.p[2] = rho * rt1 * (1.0 - sumsq + v2d2cs4 + vi); //correct
         p_eq.p[3] = rho * rt1 * (1.0 - sumsq + u2d2cs4 - ui); //correct
         p_eq.p[4] = rho * rt1 * (1.0 - sumsq + v2d2cs4 - vi); //correct
         p_eq.p[5] = rho * rt2 * (1.0 + sumsq2 + ui + vi + uv); //correct
         p_eq.p[6] = rho * rt2 * (1.0 + sumsq2 - ui + vi - uv); //correct
         p_eq.p[7] = rho * rt2 * (1.0 + sumsq2 - ui - vi + uv); //correct
         p_eq.p[8] = rho * rt2 * (1.0 + sumsq2 + ui - vi - uv); //correct

         // source term
         //R.p[i] = tauR*(ff_body*(cix - u) + ff_buoy*(ciy - v))/T*p_eq.p[i];
         //tauR = (1-1/(2*tau))
         tauRdrho = cs2i*tauR/rho;
         Rterm = -ff_bodyrho*u-ff_buoy*v;
         /*
         R.p[0] = tauRdrho*(ff_bodyrho*(0. - u) + ff_buoy*(0. - v))*p_eq.p[0]; //to be optimised
         R.p[1] = tauRdrho*(ff_bodyrho*(1. - u) + ff_buoy*(0. - v))*p_eq.p[1];
         R.p[2] = tauRdrho*(ff_bodyrho*(0. - u) + ff_buoy*(1. - v))*p_eq.p[2];
         R.p[3] = tauRdrho*(ff_bodyrho*(-1. - u) + ff_buoy*(0. - v))*p_eq.p[3];
         R.p[4] = tauRdrho*(ff_bodyrho*(0. - u) + ff_buoy*(-1. - v))*p_eq.p[4];
         R.p[5] = tauRdrho*(ff_bodyrho*(1. - u) + ff_buoy*(1. - v))*p_eq.p[5];
         R.p[6] = tauRdrho*(ff_bodyrho*(-1. - u) + ff_buoy*(1. - v))*p_eq.p[6];
         R.p[7] = tauRdrho*(ff_bodyrho*(-1. - u) + ff_buoy*(-1. - v))*p_eq.p[7];
         R.p[8] = tauRdrho*(ff_bodyrho*(1. - u) + ff_buoy*(-1. - v))*p_eq.p[8];
         */
         R.p[0] = p_eq.p[0]*tauRdrho*(Rterm); //better? not really
         R.p[1] = p_eq.p[1]*tauRdrho*(Rterm + ff_bodyrho);
         R.p[2] = p_eq.p[2]*tauRdrho*(Rterm + ff_buoy);
         R.p[3] = p_eq.p[3]*tauRdrho*(Rterm - ff_bodyrho);
         R.p[4] = p_eq.p[4]*tauRdrho*(Rterm - ff_buoy);
         R.p[5] = p_eq.p[5]*tauRdrho*(Rterm + ff_bodyrho + ff_buoy);
         R.p[6] = p_eq.p[6]*tauRdrho*(Rterm - ff_bodyrho + ff_buoy);
         R.p[7] = p_eq.p[7]*tauRdrho*(Rterm - ff_bodyrho - ff_buoy);
         R.p[8] = p_eq.p[8]*tauRdrho*(Rterm + ff_bodyrho - ff_buoy);


         pt_eq.p[0] = 0.25*T*(1+2.*u); // Is this correct? yes
         pt_eq.p[1] = 0.25*T*(1+2.*v);
         pt_eq.p[2] = 0.25*T*(1-2.*u);
         pt_eq.p[3] = 0.25*T*(1-2.*v);

         for (pp=0; pp<NPOP; pp++) {
            p[idx].p[pp] += -invtau * (p[idx].p[pp] - p_eq.p[pp]) + R.p[pp];
         }
      
         for (pp=0; pp<NPOPT; pp++) {
            pt[idx].p[pp] += -invtaut * (pt[idx].p[pp] - pt_eq.p[pp]);
         }
      }
   }
}

void bc(pop * p, popt * pt, data_t * bottemp, data_t * toptemp) {
   // applies the boundary conditions
   // #                            Top boundary (no slip tempature controlable see: toptemp)
   // #                          0,NY+1.............. NX+1,NY+1
   // #y                         .                  .
   // #^                         .                  .
   // #|noslip-wall (or periodic).      domain      . noslip-wall (or periodic)
   // #|                         .                  .
   // #|                         .                  .
   // #|                         0,0.................0,NX+1
   // #|                           bottom boundary (no slip tempature controlable see: bottemp)
   // #+-----> x

   int x, y, idx;
   int xm, xp;

   // Appling the boundary conditions on the top and bottom of the domain
   for (x=1; x<NX+1; x++) {
      xm = x-1;
      xp = x+1;

      // BC at the bottom boundary
      idx = IDX(1, x); // IDX(j,i) (j + NYP2*i)
      p[IDX(0, x)].p[2] = p[idx].p[4];
      p[IDX(0, xm)].p[5] = p[idx].p[7];
      p[IDX(0, xp)].p[6] = p[idx].p[8];
      pt[IDX(0, x)].p[1]    = bottemp[x]*0.25;// temperature T = sum(qi) approx 4*pt -> pt = T/4 = T*0.25
      
      // BC at the top boundary
      idx = IDX(NY, x);
      p[IDX(NY+1,  x)].p[4] = p[idx].p[2]; 
      p[IDX(NY+1, xm)].p[8] = p[idx].p[6];
      p[IDX(NY+1, xp)].p[7] = p[idx].p[5];
      pt[IDX(NY+1, x)].p[3] = toptemp[x]*0.25;//QTOP0;//pt[IDX(NY+1, x)].p[3] = pt[idx].p[1];
   }

   // periodic boundary conditions for left/right boundaries
   #ifdef PERIODIC
      int idxl,idxr;
      for (y=1; y < NY+1; y++) // periodic boundary y on the middle
      {
         idxl = IDX(y,0); // to this
         idxr = IDX(y,NX); // from this
         p[idxl].p[1] = p[idxr].p[1];
         p[idxl].p[5] = p[idxr].p[5];
         p[idxl].p[8] = p[idxr].p[8];
         pt[idxl].p[0] = pt[idxr].p[0];

         idxr = IDX(y,NX+1); // to this
         idxl = IDX(y,1); // from this
         p[idxr].p[3] = p[idxl].p[3];
         p[idxr].p[6] = p[idxl].p[6];
         p[idxr].p[7] = p[idxl].p[7];
         pt[idxr].p[2] = pt[idxl].p[2];
      }
   #endif

   // walled boundary conditions for left/right boundaries
   #ifdef WALLED
      int ym,yp;
      for (y=1; y < NY+1; y++)
      {
         ym = y-1;
         yp = y+1;
         idx = IDX(y,1);
         p[IDX(ym,0)].p[5] = p[idx].p[7]; // bounce back
         p[IDX(y,0)].p[1] = p[idx].p[3];
         p[IDX(yp,0)].p[8] = p[idx].p[6];
         pt[IDX(y,0)].p[0] = pt[idx].p[2]; // bounce back for the temperature population (aka adiabatic walls)

         idx = IDX(y,NX);
         p[IDX(ym,NX+1)].p[6] = p[idx].p[8];
         p[IDX(y,NX+1)].p[3] = p[idx].p[1];
         p[IDX(yp,NX+1)].p[7] = p[idx].p[5];
         pt[IDX(y,NX+1)].p[2] = pt[idx].p[0];
      }
   #endif
}