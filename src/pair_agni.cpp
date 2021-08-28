/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_agni.h"
#include "pair_meam_spline.h"
#include "atom.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "citeme.h"
#include "math_special.h"
#include "math_const.h"
#include <iostream>
#include "update.h"
using namespace std;   
using namespace LAMMPS_NS;
using namespace MathSpecial;

static const char cite_pair_agni[] =
  "pair agni command:\n\n"
  "@article{botu2015adaptive,\n"
  " author    = {Botu, Venkatesh and Ramprasad, Rampi},\n"
  " title     = {Adaptive machine learning framework to"
                " accelerate ab initio molecular dynamics},\n"
  " journal   = {International Journal of Quantum Chemistry},\n"
  " volume    = {115},\n"
  " number    = {16},\n"
  " pages     = {1074--1083},\n"
  " year      = {2015},\n"
  " publisher = {Wiley Online Library}\n"
  "}\n\n"
  "@article{botu2015learning,\n"
  " author    = {Botu, Venkatesh and Ramprasad, Rampi},\n"
  " title     = {Learning scheme to predict atomic forces"
                " and accelerate materials simulations},\n"
  " journal   = {Physical Review B},\n"
  " volume    = {92},\n"
  " number    = {9},\n"
  " pages     = {094306},\n"
  " year      = {2015},\n"
  " publisher = {APS}\n"
  "}\n\n";

#define AGNI_VERSION 2
#define MAXLINE 10240
#define MAXWORD 200  // change from 40 to 200 by YY

#define flag_2b 1
#define flag_3b 1
#define flag_mb 0 
#define mycol  121

/* ---------------------------------------------------------------------- */

PairAGNI::PairAGNI(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  nelements = 0;
  elements = NULL;
  elem2param = NULL;
  nparams = 0;
  params = NULL;
  map = NULL;
  rho_v=NULL;
  twoBodyInfo = NULL;
  pe_arr = NULL;
  pe_arr_len = 0;
  nmax = 0;
  maxNeighbors = 0;
  comm_forward = 8;
  comm_reverse = 0;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
   ------------------------------------------------------------------------- */

PairAGNI::~PairAGNI()
{
  //printf("Free: start"); 
  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;
  if (params) {
    for (int i = 0; i < nparams; ++i) {
      int n = params[i].numeta*nparams; 
      int m = params[i].numepsilon*(nparams+1)*nparams;
      for (int j = 0; j < mycol; ++j) {
        delete [] params[i].xU[j];
      }
     
      delete [] params[i].epeta; 
      delete [] params[i].eta;
      delete [] params[i].epsilon; 
      delete [] params[i].Rho;
      delete [] params[i].xU;
      delete [] params[i].yU;
    }
    memory->destroy(params);
    params = NULL;
  }
  
  delete[] twoBodyInfo;
  memory->destroy(rho_v);
 
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }
}

/* ---------------------------------------------------------------------- */

void PairAGNI::compute(int eflag, int vflag)
{
  int i,j,k,m,ii,jj,mm,inum,jnum,mnum,itype,jtype,mtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,delx2,dely2,delz2;
  double rsq,rsq2;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 
        eflag_global = vflag_global = eflag_atom = vflag_atom = 0;
  if (atom->nmax > nmax) {
    memory->destroy(rho_v);
    nmax = atom->nmax;
    memory->create(rho_v,nmax,8,"pair:rho_v");
  }
  double **x = atom->x;
  double **f = atom->f;
  double lx = domain->xprd;
  double ly = domain->yprd;
  double lz = domain->zprd;
  int *type = atom->type;
  inum = listfull->inum;
  int *ilist = listfull->ilist;
  int *numneigh = listfull->numneigh;
  int **firstneigh = listfull->firstneigh;

  int newMaxNeighbors = 0;
  for(int ii = 0; ii < inum; ii++) {
    int jnum = numneigh[ilist[ii]];
    if(jnum > newMaxNeighbors) newMaxNeighbors = jnum;
  }

  if(newMaxNeighbors > maxNeighbors) {
    maxNeighbors = newMaxNeighbors;
    delete[] twoBodyInfo;
    twoBodyInfo = new MEAM2Body[maxNeighbors];
  }  
  int nlocal = atom->nlocal;
  bool newton_pair = force->newton_pair;
  double fxtmp,fytmp,fztmp;
  int YY_cn = mycol;
  double **ff1, **ff2, **ff3;
  ff1 = new double*[inum];
  for(int myj=0; myj<inum;myj++)
  {
    ff1[myj] = new double[YY_cn];
  }

  ff2 = new double*[inum];
  for(int myj=0; myj<inum;myj++)
  {
    ff2[myj] = new double[YY_cn];
  }

  ff3 = new double*[inum];
  for(int myj=0; myj<inum;myj++)
  {
    ff3[myj] = new double[YY_cn];
  }

  for(int myj=0; myj<inum; myj++)
   {
     for(int myi= 0;myi<YY_cn;myi++)
     {
        ff1[myj][myi]=0.0;
        ff2[myj][myi]=0.0;
        ff3[myj][myi]=0.0;
     }
   }
  

  for (ii = 0; ii < inum; ii++) {
	for (int myj=0;myj<YY_cn;myj++)
	{
	  vv[myj]=0.0;
	}  
    i = ilist[ii];
    for(int k=0;k<8;k++)
           rho_v[i][k] = 0.0;   
 
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;
    const Param &iparam = params[elem2param[itype]];
    vv[(0+1)*(CN1)-1]=1.0;
    int *jlist = firstneigh[i];
    jnum = numneigh[i];
    int numBonds = 0;
    MEAM2Body* nextTwoBodyInfo = twoBodyInfo;
    double PotEg = 0.0;
     for (jj = 0; jj < jnum; jj++) 
     {
      j = jlist[jj];  
      j &= NEIGHMASK;
      jtype = map[type[j]];
      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx*delx + dely*dely + delz*delz;
      const double r = sqrt(rsq);
      const double wX = delx/r;
      const double wY = dely/r;
      const double wZ = delz/r;
      const double volume = 1.0;  
      pppp = 0.0;
      int cc=getO(itype,jtype,nparams);

	if ((rsq > 0.0) && (rsq < iparam.cutsq)) {
        if(cc==1&&r<2.10)  {pppp += LJ_Epair(r,cc);
                     continue;}
        else if (cc==0&&r<3.1099) { pppp += LJ_Epair(r,cc);continue;}
        else if (cc==2&&r<3.22999) { pppp += LJ_Epair(r,cc);continue;}

        if (flag_2b>0){ 
       
        const double cF = 0.5*(cos((MathConst::MY_PI*r)/iparam.cut)+1.0);
	for (k = 0; k < iparam.numeta; k++)
        {
        const double kkyy=1.0+(double)k;
        const double eeta=iparam.eta[k];
        const double e = exp(-r/eeta)*cos(kkyy*r);
        rho_v[i][k] +=exp(-r/iparam.eta[1])*(cos(kkyy*r)+1.0)*cF;
        vv[k+iparam.numeta*cc+0*CN1]  += cF*e;	
        }
        const double gramma=10.0;
        nextTwoBodyInfo->tag = j;
        nextTwoBodyInfo->r = r;
        if(r<iparam.cut2){
        nextTwoBodyInfo->fcut = 1.0+(gramma*r/iparam.cut2-gramma -1.0)*pow(r/iparam.cut2,gramma);
        nextTwoBodyInfo->fcut_dev =  gramma*(gramma+1.0)*pow(r/iparam.cut2,gramma-1.0)*(-1.0+r/iparam.cut2)/iparam.cut2; 
        }
        else
        {
        nextTwoBodyInfo->fcut = 0;
        nextTwoBodyInfo->fcut_dev = 0;
        }
        nextTwoBodyInfo->del[0] = wX; 
        nextTwoBodyInfo->del[1] = wY;
        nextTwoBodyInfo->del[2] = wZ;
	}
        numBonds++;     
        nextTwoBodyInfo++;
        } 
	}
        if (flag_3b>0) {
           for (int jjj = 0; jjj < numBonds; jjj++) 
           {
           const MEAM2Body bondj = twoBodyInfo[jjj];
           double r = bondj.r;
           j = bondj.tag;
           j &= NEIGHMASK;

           jtype = map[type[j]];
           MEAM2Body const* bondk = twoBodyInfo;
           for (mm = 0; mm < numBonds; mm++, ++bondk)
              {
                  if (mm==jjj) continue;
                  m = bondk->tag;
                  m &= NEIGHMASK;
                  mtype = map[type[m]];
                  const double cos3b = (bondk->del[0]*bondj.del[0] +
                                        bondk->del[1]*bondj.del[1] +
                                        bondk->del[2]*bondj.del[2]);
                  double fkk_cut=bondk->fcut*bondj.fcut;
                  int ab;
                  ab= getOrder(jtype,mtype,nparams);
                  for (int ke = 0; ke < iparam.numeta/2; ke++){
                       const int C_num1 = (ke+4*itype)+(nparams*(nparams+1)/2*iparam.numeta)+(ab*3*iparam.numeta)+(0*CN1);
                       const int C_num2 = (ke+4*itype+1*iparam.numeta)+(nparams*(nparams+1)/2*iparam.numeta)+(ab*3*iparam.numeta)+(0*CN1);
                       const int C_num3 = (ke+4*itype+2*iparam.numeta)+(nparams*(nparams+1)/2*iparam.numeta)+(ab*3*iparam.numeta)+(0*CN1);                     
                       const double angle1 = cos3b*cos3b*cos3b;
                       const double angle2 = (cos3b*cos3b-4.0*pow(cos3b,4.0));
                       const double angle3 = (1.0-4.0/3.0*cos3b*cos3b);
                       const double ratio =  fkk_cut*exp(-1.0*(pow(bondj.r,2.0)+pow(bondk->r,2.0))/(iparam.eta[2*ke]*iparam.eta[2*ke]));
                       vv[C_num1] += angle1*ratio;
                       vv[C_num2] += angle2*ratio;
                       vv[C_num3] += angle3*ratio;
                     }   
                  
              }
            
           } 
  
        } 


        if(flag_mb>0){
        for(int k=0;k<8;k++)
        {
           double lastp = rho_v[i][k];
           rho_v[i][k] = log(rho_v[i][k]);
           vv[(0+1)*(CN1)-iparam.numeta-nparams+k] = lastp*rho_v[i][k];
         }
         }

        for(int mmww=0;mmww<mycol-1;mmww++)
        {
	  pppp+=my_w[mmww]*(vv[mmww]-YY_means[mmww])*YY_devs[mmww];
	}  
 
 	pppp += cconst+my_w[mycol-1];
 
        if (eflag)
	{
       		if (eflag_global) eng_vdwl += pppp;
       	   	if (eflag_atom) eatom[i] += pppp;
	}
    if (flag_3b>0) { 
	 
	for (jj = 0; jj < numBonds; jj++) {
        const MEAM2Body bondj = twoBodyInfo[jj];
        double r = bondj.r;
        j = bondj.tag;
        j &= NEIGHMASK;
       
        jtype = map[type[j]];
	double Fj[3],Fk[3];	
        MEAM2Body const* bondk = twoBodyInfo;	
	for(mm = 0; mm < numBonds; mm++,++bondk)
        {
        if (mm==jj) continue;
        m = bondk->tag;
        m &= NEIGHMASK;  
	mtype = map[type[m]];
        const double r2 = bondk->r;
        const double cF2= bondk->fcut;
        const double cF1= bondj.fcut;
        const double dcF1= bondj.fcut_dev;
        const double dcF2= bondk->fcut_dev;
        const double cos3b = (bondj.del[0]*bondk->del[0] +
                              bondj.del[1]*bondk->del[1] +
                              bondj.del[2]*bondk->del[2]);
	int ab;
	ab= getOrder(jtype,mtype,nparams);
        Fj[0]=Fj[1]=Fj[2]=0.0;
        Fk[0]=Fk[1]=Fk[2]=0.0;		
	for (int ke = 0; ke < iparam.numeta/2; ke++){
	const int C_num1 = (ke+4*itype)+(nparams*(nparams+1)/2*iparam.numeta)+(ab*3*iparam.numeta)+(0*CN1);
        const int C_num2 = (ke+4*itype+1*iparam.numeta)+(nparams*(nparams+1)/2*iparam.numeta)+(ab*3*iparam.numeta)+(0*CN1);
        const int C_num3 = (ke+4*itype+2*iparam.numeta)+(nparams*(nparams+1)/2*iparam.numeta)+(ab*3*iparam.numeta)+(0*CN1);
        double angle1 = cos3b*cos3b*cos3b;
        double angle2 = (cos3b*cos3b-4.0*pow(cos3b,4));
        const double angle3 = (1-4.0/3.0*pow(cos3b,2));
        const double ratio =  exp(-1.0*(r*r+r2*r2)/(iparam.eta[ke*2]*iparam.eta[ke*2]));
        const double p11 = ratio*cF1*cF2*(3.0*cos3b*cos3b);
	const double p12 = angle1*cF1*cF2*ratio/(iparam.eta[ke*2]*iparam.eta[ke*2])*(-2.0); 
	const double p13 = angle1*ratio*cF2*dcF1;
	const double p14 = angle1*ratio*cF1*dcF2;
        const double p21 = ratio*cF1*cF2*1.0*(2.0*cos3b-16.0*pow(cos3b,3.0));
	const double p22 = angle2*cF1*cF2*ratio/(iparam.eta[ke*2]*iparam.eta[ke*2])*(-2.0); 
	const double p23 = angle2*ratio*cF2*dcF1;
	const double p24 = angle2*ratio*cF1*dcF2;
        const double p31 = ratio*cF1*cF2*(-8.0/3.0*pow(cos3b,1));
        const double p32 = angle3*cF1*cF2*ratio/(iparam.eta[ke*2]*iparam.eta[ke*2])*(-2.0);
        const double p33 = angle3*ratio*cF2*dcF1;
        const double p34 = angle3*ratio*cF1*dcF2;
        const double dcos1 = 1.0/r*(bondk->del[0]-cos3b*bondj.del[0])+1.0/r2*(bondj.del[0]-cos3b*bondk->del[0]);
	const double dcos2 = 1.0/r*(bondk->del[1]-cos3b*bondj.del[1])+1.0/r2*(bondj.del[1]-cos3b*bondk->del[1]);
	const double dcos3 = 1.0/r*(bondk->del[2]-cos3b*bondj.del[2])+1.0/r2*(bondj.del[2]-cos3b*bondk->del[2]);
        const double dcosj1 = 1.0/r*(bondk->del[0]-cos3b*bondj.del[0]);
	const double dcosj2 = 1.0/r*(bondk->del[1]-cos3b*bondj.del[1]);
	const double dcosj3 = 1.0/r*(bondk->del[2]-cos3b*bondj.del[2]);
        const double dcosk1 = 1.0/r2*(bondj.del[0]-cos3b*bondk->del[0]);
	const double dcosk2 = 1.0/r2*(bondj.del[1]-cos3b*bondk->del[1]);
	const double dcosk3 = 1.0/r2*(bondj.del[2]-cos3b*bondk->del[2]);
        Fj[0] -= my_w[C_num1+0*CN1] * (p11*dcosj1+p12*bondj.del[0]*bondj.r+p13*bondj.del[0]) * YY_devs[C_num1+0*CN1];
        Fj[1] -= my_w[C_num1+0*CN1] * (p11*dcosj2+p12*bondj.del[1]*bondj.r+p13*bondj.del[1]) * YY_devs[C_num1+0*CN1];
        Fj[2] -= my_w[C_num1+0*CN1] * (p11*dcosj3+p12*bondj.del[2]*bondj.r+p13*bondj.del[2]) * YY_devs[C_num1+0*CN1];
	Fk[0] -= my_w[C_num1+0*CN1] * (p11*dcosk1+p12*bondk->del[0]*bondk->r+p14*bondk->del[0]) * YY_devs[C_num1+0*CN1];
	Fk[1] -= my_w[C_num1+0*CN1] * (p11*dcosk2+p12*bondk->del[1]*bondk->r+p14*bondk->del[1]) * YY_devs[C_num1+0*CN1];
	Fk[2] -= my_w[C_num1+0*CN1] * (p11*dcosk3+p12*bondk->del[2]*bondk->r+p14*bondk->del[2]) * YY_devs[C_num1+0*CN1];
	Fj[0] -= my_w[C_num2+0*CN1] * (p21*dcosj1+p22*bondj.del[0]*bondj.r+p23*bondj.del[0]) * YY_devs[C_num2+0*CN1];
        Fj[1] -= my_w[C_num2+0*CN1] * (p21*dcosj2+p22*bondj.del[1]*bondj.r+p23*bondj.del[1]) * YY_devs[C_num2+0*CN1];
        Fj[2] -= my_w[C_num2+0*CN1] * (p21*dcosj3+p22*bondj.del[2]*bondj.r+p23*bondj.del[2]) * YY_devs[C_num2+0*CN1];	
	Fk[0] -= my_w[C_num2+0*CN1] * (p21*dcosk1+p22*bondk->del[0]*bondk->r+p24*bondk->del[0]) * YY_devs[C_num2+0*CN1];
	Fk[1] -= my_w[C_num2+0*CN1] * (p21*dcosk2+p22*bondk->del[1]*bondk->r+p24*bondk->del[1]) * YY_devs[C_num2+0*CN1];
	Fk[2] -= my_w[C_num2+0*CN1] * (p21*dcosk3+p22*bondk->del[2]*bondk->r+p24*bondk->del[2]) * YY_devs[C_num2+0*CN1];	
        Fj[0] -= my_w[C_num3+0*CN1] * (p31*dcosj1+p32*bondj.del[0]*bondj.r+p33*bondj.del[0]) * YY_devs[C_num3+0*CN1];
        Fj[1] -= my_w[C_num3+0*CN1] * (p31*dcosj2+p32*bondj.del[1]*bondj.r+p33*bondj.del[1]) * YY_devs[C_num3+0*CN1];
        Fj[2] -= my_w[C_num3+0*CN1] * (p31*dcosj3+p32*bondj.del[2]*bondj.r+p33*bondj.del[2]) * YY_devs[C_num3+0*CN1];
        Fk[0] -= my_w[C_num3+0*CN1] * (p31*dcosk1+p32*bondk->del[0]*bondk->r+p34*bondk->del[0]) * YY_devs[C_num3+0*CN1];
        Fk[1] -= my_w[C_num3+0*CN1] * (p31*dcosk2+p32*bondk->del[1]*bondk->r+p34*bondk->del[1]) * YY_devs[C_num3+0*CN1];
        Fk[2] -= my_w[C_num3+0*CN1] * (p31*dcosk3+p32*bondk->del[2]*bondk->r+p34*bondk->del[2]) * YY_devs[C_num3+0*CN1];
	} 
	f[i][0] -= Fj[0]+Fk[0];
	f[i][1] -= Fj[1]+Fk[1];
	f[i][2] -= Fj[2]+Fk[2];
	f[j][0] += Fj[0];   
	f[j][1] += Fj[1];
	f[j][2] += Fj[2];
	f[m][0] += Fk[0]; 
	f[m][1] += Fk[1];
	f[m][2] += Fk[2];
	if(evflag) 
	{
	    double dj[3],dk[3];
            dj[0]=-bondj.del[0]*bondj.r;
	    dj[1]=-bondj.del[1]*bondj.r;
	    dj[2]=-bondj.del[2]*bondj.r;		
	    dk[0]=-bondk->del[0]*bondk->r;
	    dk[1]=-bondk->del[1]*bondk->r;
	    dk[2]=-bondk->del[2]*bondk->r;
            ev_tally3(i, j, m, 0.0, 0.0, Fj, Fk, dj, dk);
	}
	} 
	}
    }
   }
   comm->forward_comm_pair(this);
    
    for (ii = 0; ii < inum; ii++) { 
         i = ilist[ii];
         itype = map[type[i]];
         xtmp = x[i][0];
         ytmp = x[i][1];
         ztmp = x[i][2];
         const Param &iparam = params[elem2param[itype]];
         int *jlist = firstneigh[i];
         jnum = numneigh[i];
         for(int jj = 0; jj < jnum; jj++) { 
              j = jlist[jj];
              j &= NEIGHMASK;
              jtype = map[type[j]];
              delx = x[j][0] - xtmp;
              dely = x[j][1] - ytmp;
              delz = x[j][2] - ztmp;
              rsq = delx*delx + dely*dely + delz*delz;
              double r = sqrt(rsq);
              const double volume = 1.0;  
               int cc=getO(itype,jtype,nparams);
         if (flag_2b>0) {  
	       if ((rsq > 0.0) && (rsq < iparam.cutsq)) {
                   
              
	            double fpair=0.0;	
                    
            
                    const double cF = 0.5*(cos((MathConst::MY_PI*r)/iparam.cut)+1.0);
	            const double dcF = -0.5*(MathConst::MY_PI)/iparam.cut*(sin(MathConst::MY_PI*r/iparam.cut));
	            
                    if(cc==1&&r<2.12)
                        fpair=LJ_fpair(r,cc);
                    else if (cc==0&&r<3.12)
                        fpair=LJ_fpair(r,cc);
                    else if (cc==2&&r<3.25)
                        fpair=LJ_fpair(r,cc);
                    else{
                    for (k = 0; k < iparam.numeta; k++) {  
                        const double kkyy=1.0+(double)k;
                        const double eeta=iparam.eta[k];
                        const double e = exp(-r/eeta); 
                        const double t2b = (-kkyy*sin(kkyy*r)*cF-(1.0/eeta)*cos(kkyy*r)*cF+cos(kkyy*r)*dcF)*exp(-r/eeta);
                        fpair += t2b/r*my_w[k+iparam.numeta*cc+0*CN1]*YY_devs[k+iparam.numeta*cc+0*CN1];
                        if(flag_mb>0) { 
                           double tmb = (-k*sin(k*r)*cF-(1/iparam.eta[1])*(cos(k*r)+1)*cF+(1+cos(k*r))*dcF)*exp(-r/iparam.eta[1]);
                           fpair +=1/r*tmb*my_w[(1)*CN1-iparam.numeta-nparams+k]*(1.0+rho_v[i][k]);
                        }
                    }  
                    }

                    f[i][0] += delx * fpair;
                    f[i][1] += dely * fpair;
                    f[i][2] += delz * fpair;
                    f[j][0] -= delx * fpair;
                    f[j][1] -= dely * fpair;
                    f[j][2] -= delz * fpair;
                    if (evflag) ev_tally_full(i,0.0, 0.0, -fpair, delx, dely, delz);
             } 
	  } 
          } 
    }  
    for(i=0;i<inum;i++)
    {
      delete []ff1[i];
    }
    delete [] ff1;

    for(i=0;i<inum;i++)
    {
      delete []ff2[i];
    }
    delete [] ff2;

    for(i=0;i<inum;i++)
    {
      delete []ff3[i];
    }
    delete [] ff3;
   if (vflag_fdotr) virial_fdotr_compute();	
   }
 /* ---------------------------------------------------------------------- */

void PairAGNI::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void PairAGNI::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   ------------------------------------------------------------------------- */

void PairAGNI::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
   
  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }
  read_file(arg[2]);
  setup_params();
  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;
  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ---------------------------------------------------------------------- */

void PairAGNI::read_file(char *file)
{
  memory->sfree(params);
  params = NULL;
  nparams = 0;
  
  FILE *fp;
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open AGNI potential file %s",file);
      error->one(FLERR,str);
    }
  }

  int i,j,n,nwords,curparam,wantdata;
  char line[MAXLINE],*ptr;
  int eof = 0;
  char **words = new char*[MAXWORD+1];

  while (1) {
    n = 0;
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
	
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    if (nwords > MAXWORD)
      error->all(FLERR,"Increase MAXWORD and recompile");

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;
    --nwords;

    if ((nwords == 2) && (strcmp(words[0],"generation") == 0)) {
      int ver = atoi(words[1]);
      if (ver != AGNI_VERSION)
        error->all(FLERR,"Incompatible AGNI potential file version");
      if ((ver == 1) && (nelements != 1))
        error->all(FLERR,"Cannot handle multi-element systems with this potential");

    } else if ((nwords == 2) && (strcmp(words[0],"n_elements") == 0)) {
      nparams = atoi(words[1]);     
      if ((nparams < 1) || params) 
        error->all(FLERR,"Invalid AGNI potential file");
      params = memory->create(params,nparams,"pair:params");
      memset(params,0,nparams*sizeof(Param));
      curparam = -1;

    } else if ((nwords == 3) && (strcmp(words[0],"E_factor") == 0)){
	  Emax = atof(words[1]);
	  Emin = atof(words[2]);
	  
	} else if ((nwords == 4) && (strcmp(words[0],"S_factor") == 0)){
	  int YY_f;
      YY_f = atoi(words[1]);
      Smax[YY_f] = atof(words[2]);
      Smin[YY_f] = atof(words[3]);	  
	} else if ((nwords == 4) && (strcmp(words[0],"F_factor") == 0)){
	  int YY_f;
      YY_f = atoi(words[1]);
      Fmax[YY_f] = atof(words[2]);
      Fmin[YY_f] = atof(words[3]);	
	} else if ((nwords == 2) && (strcmp(words[0],"N_flag") == 0)){
	  N_flag = atoi(words[1]);
	}
	
	
	else if (params && (nwords == nparams+1) && (strcmp(words[0],"element") == 0)) {
      wantdata = -1;
      for (i = 0; i < nparams; ++i) {
        for (j = 0; j < nelements; ++j)
          if (strcmp(words[i+1],elements[j]) == 0) break;
        if (j == nelements)
          error->all(FLERR,"No suitable parameters for requested element found");
        else params[i].ielement = j;
      }
    } else if (params && (nwords == 2) && (strcmp(words[0],"interaction") == 0)) {
      for (i = 0; i < nparams; ++i)
        if (strcmp(words[1],elements[params[i].ielement]) == 0) curparam = i;
    } else if ((curparam >=0) && (nwords == 1) && (strcmp(words[0],"endVar") == 0)) {
      int numtrain = params[curparam].numtrain;
      int numeta = params[curparam].numeta;
      int numepsilon = params[curparam].numepsilon;
      int num_column = mycol;
      params[curparam].yU = new double[numtrain];
      params[curparam].xU = new double*[num_column];
      for (i = 0; i < num_column; ++i)
        params[curparam].xU[i] = new double[numtrain];

      wantdata = curparam;
      curparam = -1;
    } 
      else if ((curparam >=0) && (nwords == 2) && (strcmp(words[0],"C") == 0)) {
      cconst =atof(words[1]);
    } 
      else if ((curparam >=0) && (nwords == 2) && (strcmp(words[0],"Rc") == 0)) {
      params[curparam].cut = atof(words[1]);
    } else if ((curparam >=0) && (nwords == 2) && (strcmp(words[0],"Rs") == 0)) {
      params[curparam].cut2 = atof(words[1]); 
    } else if ((curparam >=0) && (nwords == 2) && (strcmp(words[0],"Rm") == 0)) {
      params[curparam].cut3 = atof(words[1]); 
    } 
      else if ((curparam >=0) && (nwords > 1) && (strcmp(words[0],"Mu") == 0)) {
      params[curparam].numRho = nwords-1;
      params[curparam].Rho = new double[nwords-1];
      for (i = 0, j = 1 ; j < nwords; ++i, ++j)
        params[curparam].Rho[i] = atof(words[j]);
    }
       else if ((curparam >=0) && (nwords == 3) && (strcmp(words[0],"eta") == 0)) {
      params[curparam].numeta = atoi(words[1]);
      params[curparam].eta = new double[atoi(words[1])];
      for (i = 0 ; i < atoi(words[1]); ++i)
        {params[curparam].eta[i] = pow(atof(words[2]),(double)(i));
        }

    } else if ((curparam >=0) && (nwords == mycol+1) && (strcmp(words[0],"alpha") == 0)) {
      for (i = 0, j = 1 ; j < nwords; ++i, ++j)
        my_w[i] = atof(words[j]);
      
    } else if ((curparam >=0) && (nwords == mycol+1) && (strcmp(words[0],"Std_dev") == 0)) {
      for (i = 0, j = 1 ; j < nwords; ++i, ++j)
        YY_devs[i] = atof(words[j]);

    } else if ((curparam >=0) && (nwords == mycol+1) && (strcmp(words[0],"mean") == 0)) {
      for (i = 0, j = 1 ; j < nwords; ++i, ++j)
        YY_means[i] = atof(words[j]);

    }
   else {
      if (comm->me == 0)
        error->warning(FLERR,"Ignoring unknown content in AGNI potential file.");
    }
  }
  delete [] words;
}

/* ---------------------------------------------------------------------- */

void PairAGNI::setup_params()
{
  int i,m,n;
  double rtmp, rtmp2,rtmp3; 

  memory->destroy(elem2param);
  memory->create(elem2param,nelements,"pair:elem2param");

  for (i = 0; i < nelements; i++) {
    n = -1;
    for (m = 0; m < nparams; m++) {
      if (i == params[m].ielement) {
        if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
        n = m;
      }
    }
    if (n < 0) error->all(FLERR,"Potential file is missing an entry");
    elem2param[i] = n;
  }

  // compute parameter values derived from inputs

  // set cutsq using shortcut to reduce neighbor list for accelerated
  // calculations. cut must remain unchanged as it is a potential parameter
  // (cut = a*sigma)

  cutmax = 0.0;
  for (m = 0; m < nparams; m++) {
    rtmp = params[m].cut;
    rtmp2 = params[m].cut2;
    rtmp3 = params[m].cut3;
    
    params[m].cutsq = rtmp * rtmp;
    params[m].cutsq2 = rtmp2 * rtmp2;
    params[m].cutsq3 = rtmp3 * rtmp3;
    if (rtmp > cutmax) cutmax = rtmp;
  }
}

int PairAGNI::getOrder(int i, int j, int m)
{
   if (m==1) 
	{return 0;}
   else{
  int order[m][m];
  int flag;
  flag = 0;
  for(int z=0;z<m;z++)
    for(int y=0;y<m;y++)
      {
	order[z][y]=flag;
	flag++;
      }
   return order[i][j];}
	
}


int PairAGNI::getO(int i, int j, int m)
{
   if (m==1) 
        {return 0;}
   else{
  int order[m][m];
  int flag;
  flag = 0;
  for(int z=0;z<m;z++)
    for(int y=z;y<m;y++)
      {
        order[z][y]=flag;
        order[y][z]=flag;
        flag++;
      }
   return order[i][j];}
 
}




double PairAGNI::myCos(double a1,double b1,double c1,double a2,double b2,double c2)
{
  double dot_value=a1*a2+b1*b2+c1*c2;
  double length=sqrt(a1*a1+b1*b1+c1*c1)*sqrt(a2*a2+b2*b2+c2*c2);
	
  return dot_value/length;
	
}	




/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */
void PairAGNI::init_style()
{
        if(force->newton_pair == 0)
                error->all(FLERR,"Pair style meam/spline requires newton pair on");

        // Need both full and half neighbor list.
        int irequest_full = neighbor->request(this);
        neighbor->requests[irequest_full]->id = 1;
        neighbor->requests[irequest_full]->half = 0;
        neighbor->requests[irequest_full]->full = 1;
        int irequest_half = neighbor->request(this);
        neighbor->requests[irequest_half]->id = 2;
        neighbor->requests[irequest_half]->half = 0;
        neighbor->requests[irequest_half]->half_from_full = 1;
        neighbor->requests[irequest_half]->otherlist = irequest_full;
		
  
	
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   half or full
------------------------------------------------------------------------- */
void PairAGNI::init_list(int id, NeighList *ptr)
{
        if(id == 1) listfull = ptr;
        else if(id == 2) listhalf = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
double PairAGNI::init_one(int i, int j)
{
        //return cutoff;
		return cutmax;
}

void PairAGNI::grab(FILE *fptr, int n, double *list)
{
  char *ptr;
  char line[MAXLINE];

  int i = 0;
  while (i < n) {
    fgets(line,MAXLINE,fptr);
    ptr = strtok(line," \t\n\r\f");
    list[i++] = atof(ptr);
    while (ptr = strtok(NULL," \t\n\r\f")) list[i++] = atof(ptr);
  }
}

double PairAGNI::LJ_fpair(double r, int c)
{

   double value;
   if (c==1)
   {
    value = 760.0*pow(r,-5);
    value += -2.804*exp(8.0/r)*pow(r,-2);//-760+2.804*exp(8);
    return 0.5*value/r;
   }
   else if (c==0)
  {
    value = (-4)*2.478*pow(r,-5)*exp(8.0/r);
    value += -2.478*pow(r,-4)*exp(8.0/r)*pow(r,-2);
    return 0.5*value/r;
  }
  else if (c==2)
  {
    value = (-4)*9.125*pow(r,-5)*exp(8.0/r);
    value += -9.125*pow(r,-4)*exp(8.0/r)*pow(r,-2);
    return 0.5*value/r;
  }
}

double PairAGNI::LJ_Epair(double r,int c)
{
  double value;
  if  (c==1)
  {
    value = -190*pow(r,-4);
    value += 0.2603*exp(8.0/r)-2.889235960045479;  
    return 0.5*value;
  }
  
  else if (c==0)
  {
    value = 2.478*pow(r,-4)*exp(8.0/r);
    value += -0.016711504847934044; 
    return 0.5*value;
  }
  else if (c==2)
  {
    value = 9.125*pow(r,-4)*exp(8.0/r);
    value += -0.8061894875211051;  
    return 0.5*value;
  }
}

/* ---------------------------------------------------------------------- */

int PairAGNI::pack_forward_comm(int n, int *list, double *buf,
                                      int pbc_flag, int *pbc)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
        for (k = 0; k < 8; k++)
          buf[m++] = rho_v[j][k];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairAGNI::unpack_forward_comm(int n, int first, double *buf)
{
    int i,k,m,last;

  m = 0;
 last = first + n;
  for (i = first; i < last; i++){
   for (k = 0; k < 8; k++)
      rho_v[i][k] = buf[m++];
    }


}

/* ---------------------------------------------------------------------- */

int PairAGNI::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;

   m = 0;
  last = first + n;
  for (i = first; i < last; i++)
         for (k = 0; k < 8; k++)
          buf[m++] = rho_v[i][k];
  return m;
}

/* ---------------------------------------------------------------------- */

void PairAGNI::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
      j = list[i];
   for (k = 0; k < 8; k++)
      rho_v[j][k] = buf[m++];
    }
}

/* ----------------------------------------------------------------------
   Returns memory usage of local atom-based arrays
------------------------------------------------------------------------- */
double PairAGNI::memory_usage()
{
        return nmax *8* sizeof(double);
}

