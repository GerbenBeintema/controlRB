

void calxyUvrhot(simulation * sim) { // uses p1
   int x,y,idx;
   data_t * uvec, * vvec, * rhovec, * temperaturevec;
   popt * pt1;
   pop * p1;
   data_t T,rho,u,v;
   for (int s=0;s<NSIM;s++) {
      uvec = sim->u + s*NXP2*NYP2;
      vvec = sim->v + s*NXP2*NYP2;
      rhovec = sim->rho + s*NXP2*NYP2;
      temperaturevec = sim->temperature + s*NXP2*NYP2;
      p1 = sim->p1+s*NXP2*NYP2;
      pt1 = sim->pt1+s*NXP2*NYP2;
      for (y = 0; y < NYP2; y++)
      {
         for (x=0;x<NXP2;x++) {
            idx = IDX(y, x);
            T = vt( pt1[idx] );
            rho = m(p1[idx]);
            u = vx(p1[idx]) / rho;
            v = vy(p1[idx]) / rho;
            uvec[idx] = u;
            vvec[idx] = v;
            rhovec[idx] = rho;
            temperaturevec[idx] = T;
         }
      }
   }
}