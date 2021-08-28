/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(agni,PairAGNI)

#else

#ifndef LMP_PAIR_AGNI_H
#define LMP_PAIR_AGNI_H

#include "pair.h"
//#include <vector>
//#include <dlib/svm.h>

using namespace std;
//using namespace dlib;

namespace LAMMPS_NS {

class PairAGNI : public Pair {
 public:
  PairAGNI(class LAMMPS *);
  virtual ~PairAGNI();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  void init_list(int, class NeighList *);
  
  void grab(FILE *, int, double *);  //YY:2018-05-03
  
  double init_one(int, int);


  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();
  struct Param {
    double cut,cutsq,cut2,cutsq2,cut3,cutsq3;      
    double *eta,**xU,*yU,*epsilon,*Rho,*epeta;  
    int numeta,numtrain,numtest,ielement,numepsilon,numRho,numepe;  
    };

 protected:
  double cutmax;                
  int nelements;         
  int N_flag;           
  double Emax,Emin;
  double Fmax[3],Fmin[3],Smax[6],Smin[6];
  double **rho_v;
  char **elements;     
  int *elem2param;    
  int *map;          
  int nparams;      
  Param *params;   
  
  double *pe_arr;
  int pe_arr_len;
  
  void read_file(char *);
  virtual void setup_params();
  

  struct MEAM2Body {
                int tag;
                double r;
                double fcut;
                double fcut_dev;
                double del[3];
        };

  int nmax;                  
  int maxNeighbors;         
  MEAM2Body* twoBodyInfo;  
  void allocate();
  double fun_cutoff(double, double);
  double fun_cutoff(double, double, double);
  double fun_cutoff_dev(double,double,double);
  double LJ_fpair(double, int );
  double LJ_Epair(double, int );
  double Hx(double );
  void   Data_Fitting();
  void   costheta_d(double,const double *, double,const double *, double, double *, double *);

  inline double vec3_dot(const double *x,const double *y) {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  }
  inline void vec3_add(const double *x, const double *y, double *z) {
    z[0] = x[0]+y[0];  z[1] = x[1]+y[1];  z[2] = x[2]+y[2];
  }
  inline void vec3_scale(const double k, const double *x, double *y) {
    y[0] = k*x[0];  y[1] = k*x[1];  y[2] = k*x[2];
  }
  inline void vec3_scaleadd(const double k,const double *x,const double *y, double *z) {
    z[0] = k*x[0]+y[0];  z[1] = k*x[1]+y[1];  z[2] = k*x[2]+y[2];
  }
  int N_type; 
  int CN1; 
  double cconst;
  double pppp;     
  double my_w[121],vv[121],fvect_dev[121],YY_means[121],YY_devs[121];
  
  int getOrder(int, int, int);
  int getO(int, int, int);
  double myCos(double,double,double,double,double,double);
  
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Cannot open AGNI potential file %s

The specified AGNI potential file cannot be opened.  Check that the path
and name are correct.

E: Incorrect format in AGNI potential file

The potential file is not compatible with the AGNI pair style
implementation in this LAMMPS version.

*/
