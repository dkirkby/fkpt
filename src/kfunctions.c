/*==============================================================================
 NAME: kfunctions.c				[code for fk - Perturbation Theory]
 Alejandro Aviles (avilescervantes@gmail.com)
 ================================================================================
*/



#include "globaldefs.h"
#include "protodefs.h"
#include "models.h"



local void quadrature(real ki);
local void k_functions(void);
//~ local void pk_non_wiggle(void);

local real Interpolation_nr(real k, double kPS[], double pPS[], int nPS, double pPS2[]);

local real get_sigma8(void);
local real sigma28_function_int(real y);

local real sigma2L_function_int(real y);
local real Sigma2_int(real y);
local real deltaSigma2_int(real y);
local real sigma2v_function_int(real y);
local real sigma_constants(void);

#define KMIN    1.0e-20
#define _KERNELS_LCDMfk_ 0

// Forward declaration for HDF5 dump
local void dump_kfunctions_hdf5(const char *filename);

global void compute_kfunctions(void)
{
    stream outstrQsRs, outtables;
    real dk;
    real bTime, t_start, t_init, t_sigma8, t_sigma_const, t_kfunc_loop;
    real kBAOmin=0.005, kBAOmax=1.0, epsquadsave;
    int iBAOmin, iBAOmax;
    global_D2v2_ptr ptmp;
    global_D3v2_ptr ptmpR1;
    real fR0save;
    //~ real sigma2psi;


    bTime = second();

    // gd.f0=f_growth_LCDM();  //Modificacion

    t_start = second();
    ptmp = DsSecondOrder_func(KMIN, KMIN, KMIN);
    KA_LCDM = DA2D2(ptmp) / ( (3.0/7.0) * Dpk1D2(ptmp) * Dpk2D2(ptmp) );
    KAp_LCDM = DA2primeD2(ptmp)
                / ( (3.0/7.0) * Dpk1D2(ptmp) * Dpk2D2(ptmp) ) -
               2.0 * DA2D2(ptmp)
                / ( (3.0/7.0) * Dpk1D2(ptmp) * Dpk2D2(ptmp) )* gd.f0;
    KB_LCDM = KA_LCDM;

    ptmpR1 = DsThirdOrder_func(0.0000001, KMIN, KMIN);
    KR1_LCDM = (21.0/5.0)*D3symmD3(ptmpR1)
        /( DpkD3(ptmpR1)*DppD3(ptmpR1)*DppD3(ptmpR1) );
    KR1p_LCDM = (21.0/5.0)*D3symmprimeD3(ptmpR1)
        /( DpkD3(ptmpR1)*DppD3(ptmpR1)*DppD3(ptmpR1) )/(3.*gd.f0);
    t_init = second() - t_start;

    t_start = second();
	get_sigma8();
	if(cmd.chatty==3) fprintf(stdout,"%g\n",gd.sigma8);
    t_sigma8 = second() - t_start;

    t_start = second();
    sigma_constants();
    t_sigma_const = second() - t_start;

    if(cmd.chatty==1){
		fprintf(stdout,"\nA_LCDM=%g, Ap_LCDM=%g, KR1_LCDM = %g, KR1p_LCDM = %g"
					,KA_LCDM, KAp_LCDM, KR1_LCDM, KR1p_LCDM);
        // fprintf(stdout,"\nsigma quadratures from kmin = %g to kmax = %g", kPS[1],kPS[nPSLT]);
		fprintf(stdout,"\ns2psi = %g,   s2v = %g,   Sigma2 = %g,   deltaSigma2 = %g",
			gd.sigma2L, gd.sigma2v, gd.Sigma2, gd.deltaSigma2);

		fprintf(stdout,"\nk-functions:");
		fprintf(stdout," Nk=%d values from kmin=%g to kmax=%g ",
            cmd.Nk, cmd.kmin, cmd.kmax);
        // fprintf(stdout,"\nf_LCDM=%g",f_growth_LCDM());
		//~ fprintf(stdout,"\nsigma8(z=%g)=%g \n",cmd.xstop,gd.sigma8);
    };

    t_start = second();
    k_functions();
    t_kfunc_loop = second() - t_start;


    if(cmd.chatty==1) {
        fprintf(stdout,"\n...time = %g seconds",second()-bTime);
        fprintf(stdout,"\n\n--- compute_kfunctions breakdown ---\n");
        fprintf(stdout,"  Initialization:     %8.3f s (%5.1f%%)\n",
                t_init, 100.0*t_init/(t_init+t_sigma8+t_sigma_const+t_kfunc_loop));
        fprintf(stdout,"  sigma8 calculation: %8.3f s (%5.1f%%)\n",
                t_sigma8, 100.0*t_sigma8/(t_init+t_sigma8+t_sigma_const+t_kfunc_loop));
        fprintf(stdout,"  sigma constants:    %8.3f s (%5.1f%%)\n",
                t_sigma_const, 100.0*t_sigma_const/(t_init+t_sigma8+t_sigma_const+t_kfunc_loop));
        fprintf(stdout,"  k-functions loop:   %8.3f s (%5.1f%%)\n",
                t_kfunc_loop, 100.0*t_kfunc_loop/(t_init+t_sigma8+t_sigma_const+t_kfunc_loop));
        fprintf(stdout,"  TOTAL:              %8.3f s\n",
                t_init+t_sigma8+t_sigma_const+t_kfunc_loop);
        fprintf(stdout,"------------------------------------\n");
    }

    // HDF5 dump if requested
    if (!strnull(cmd.dumpKfunctions)) {
        if(cmd.chatty==1) {
            fprintf(stdout,"\nDumping k-functions snapshot to HDF5 file: %s\n", cmd.dumpKfunctions);
        }
        dump_kfunctions_hdf5(cmd.dumpKfunctions);
        if(cmd.chatty==1) {
            fprintf(stdout,"HDF5 dump completed.\n");
        }
    }

}
#undef KMIN



#define _K_LOGSPACED_  1
local void k_functions(void)
{
    global_kFs qrs, qrs_nw;
    real aTime;
    real kval, ki, pkl, pkl_nw, fk;
    //~ int counter;
    int i;
    real dk;
    real t_start_k, t_per_k, t_total_loop = 0.0;
    if (_K_LOGSPACED_ ==1){
		dk = (rlog10(cmd.kmax/cmd.kmin))/((real)(cmd.Nk - 1));
	} else {
		dk = (cmd.kmax-cmd.kmin)/((real)(cmd.Nk - 1));
	}

    for (i=1; i<=cmd.Nk; i++) {
        t_start_k = second();
		if (_K_LOGSPACED_ ==1){
			kval = rlog10(cmd.kmin) + dk*((real)(i - 1));
			ki = rpow(10.0,kval);
		} else {
			ki = cmd.kmin + dk*((real)(i - 1));
		}
        qrs = ki_functions_driver(ki,kPS, pPS, nPSLT, pPS2);
        qrs_nw = ki_functions_driver(ki,kPS, pPS_nw, nPSLT, pPS2_nw);

        pkl = psInterpolation_nr(ki, kPS, pPS, nPSLT);
        fk = Interpolation_nr(ki, kPS, fkT, nPSLT, fkT2);
        pkl_nw = Interpolation_nr(ki, kPS, pPS_nw, nPSLT, pPS2_nw);


		kFArrays.kT[i-1]           =  ki ;
		kFArrays.P22ddT[i-1]       =  qrs.P22dd;
		kFArrays.P22duT[i-1]       =  qrs.P22du;
		kFArrays.P22uuT[i-1]       =  qrs.P22uu;
		// A
		kFArrays.I1udd1AT[i-1]     =  qrs.I1udd1A;
		kFArrays.I2uud1AT[i-1]     =  qrs.I2uud1A;
		kFArrays.I2uud2AT[i-1]     =  qrs.I2uud2A;
		kFArrays.I3uuu2AT[i-1]     =  qrs.I3uuu2A;
		kFArrays.I3uuu3AT[i-1]     =  qrs.I3uuu3A;
		//  B plus C
		kFArrays.I2uudd1BpCT[i-1]  =  qrs.I2uudd1BpC;
		kFArrays.I2uudd2BpCT[i-1]  =  qrs.I2uudd2BpC;
		kFArrays.I3uuud2BpCT[i-1]  =  qrs.I3uuud2BpC;
		kFArrays.I3uuud3BpCT[i-1]  =  qrs.I3uuud3BpC;
		kFArrays.I4uuuu2BpCT[i-1]  =  qrs.I4uuuu2BpC;
		kFArrays.I4uuuu3BpCT[i-1]  =  qrs.I4uuuu3BpC;
		kFArrays.I4uuuu4BpCT[i-1]  =  qrs.I4uuuu4BpC;
		//  Bias
		kFArrays.Pb1b2T[i-1]       =  qrs.Pb1b2;
		kFArrays.Pb1bs2T[i-1]      =  qrs.Pb1bs2;
		kFArrays.Pb22T[i-1]        =  qrs.Pb22;
		kFArrays.Pb2s2T[i-1]       =  qrs.Pb2s2;
		kFArrays.Ps22T[i-1]        =  qrs.Ps22;
		kFArrays.Pb2thetaT[i-1]    =  qrs.Pb2theta;
		kFArrays.Pbs2thetaT[i-1]   =  qrs.Pbs2theta;
		//
		kFArrays.P13ddT[i-1]       =  qrs.P13dd;
		kFArrays.P13duT[i-1]       =  qrs.P13du;
		kFArrays.P13uuT[i-1]       =  qrs.P13uu;
		kFArrays.sigma32PSLT[i-1]  =  qrs.sigma32PSL;
		kFArrays.pklT[i-1]         =  pkl;
		kFArrays.fkT[i-1]          =  fk;




		kFArrays_nw.kT[i-1]           =  ki ;
		kFArrays_nw.P22ddT[i-1]       =  qrs_nw.P22dd;
		kFArrays_nw.P22duT[i-1]       =  qrs_nw.P22du;
		kFArrays_nw.P22uuT[i-1]       =  qrs_nw.P22uu;
		// A
		kFArrays_nw.I1udd1AT[i-1]     =  qrs_nw.I1udd1A;
		kFArrays_nw.I2uud1AT[i-1]     =  qrs_nw.I2uud1A;
		kFArrays_nw.I2uud2AT[i-1]     =  qrs_nw.I2uud2A;
		kFArrays_nw.I3uuu2AT[i-1]     =  qrs_nw.I3uuu2A;
		kFArrays_nw.I3uuu3AT[i-1]     =  qrs_nw.I3uuu3A;
		//  B plus C
		kFArrays_nw.I2uudd1BpCT[i-1]  =  qrs_nw.I2uudd1BpC;
		kFArrays_nw.I2uudd2BpCT[i-1]  =  qrs_nw.I2uudd2BpC;
		kFArrays_nw.I3uuud2BpCT[i-1]  =  qrs_nw.I3uuud2BpC;
		kFArrays_nw.I3uuud3BpCT[i-1]  =  qrs_nw.I3uuud3BpC;
		kFArrays_nw.I4uuuu2BpCT[i-1]  =  qrs_nw.I4uuuu2BpC;
		kFArrays_nw.I4uuuu3BpCT[i-1]  =  qrs_nw.I4uuuu3BpC;
		kFArrays_nw.I4uuuu4BpCT[i-1]  =  qrs_nw.I4uuuu4BpC;
		//  Bias
		kFArrays_nw.Pb1b2T[i-1]       =  qrs_nw.Pb1b2;
		kFArrays_nw.Pb1bs2T[i-1]      =  qrs_nw.Pb1bs2;
		kFArrays_nw.Pb22T[i-1]        =  qrs_nw.Pb22;
		kFArrays_nw.Pb2s2T[i-1]       =  qrs_nw.Pb2s2;
		kFArrays_nw.Ps22T[i-1]        =  qrs_nw.Ps22;
		kFArrays_nw.Pb2thetaT[i-1]    =  qrs_nw.Pb2theta;
		kFArrays_nw.Pbs2thetaT[i-1]   =  qrs_nw.Pbs2theta;
		//
		kFArrays_nw.P13ddT[i-1]       =  qrs_nw.P13dd;
		kFArrays_nw.P13duT[i-1]       =  qrs_nw.P13du;
		kFArrays_nw.P13uuT[i-1]       =  qrs_nw.P13uu;
		kFArrays_nw.sigma32PSLT[i-1]  =  qrs_nw.sigma32PSL;
		kFArrays_nw.pklT[i-1]         =  pkl_nw;
		kFArrays_nw.fkT[i-1]          =  fk;

		//~ fprintf(stdout,"(k=%e, f=%e), ",ki,kFArrays.fkT[i-1]);

        t_per_k = second() - t_start_k;
        t_total_loop += t_per_k;

        if (cmd.chatty==1 && (i==1 || i==cmd.Nk || i%20==0)) {
            fprintf(stdout,"\n  k[%3d/%3d]: k=%9.6f, time=%7.3f ms, avg=%7.3f ms/k",
                    i, cmd.Nk, ki, 1e3*t_per_k, 1e3*t_total_loop/i);
        }
    }

    if(cmd.chatty==1) {
        fprintf(stdout,"\n  Total k-loop time: %8.3f ms, avg per k: %7.3f ms\n",
                1e3*t_total_loop, 1e3*t_total_loop/cmd.Nk);
    }
}

#undef _K_LOGSPACED_


#define QROMBERG     qromo
#define KK  5






//Modificacion-LCDMfk
// Q and R functions quadrature
// kk is the inner integration moment:
// kk = k * r, so usual notation kk = p
//~ global_kFs ki_functions(real eta, real ki)
global_kFs ki_functions(real ki, double kPKL[], double pPKL[], int nPKLT, double pPKL2[])
{
    int i, j;
    real pkl_k;
    real t_q_start, t_q_loop = 0.0, t_r_loop = 0.0;

    real PSLA, PSLB, psl;
	real fk, fp, fkmp, pklp, pklkmp;
    real rmin, rmax;
    real r, deltar, r2, y, y2;
    real mumin, mumax;
    real x, w, x2;
    real psl1;
    int Nx, nGL;
    real ypi, dk;
    real *xxGL, *wwGL, *xGL, *wGL;
    real kmin, kmax, pmin, pmax;
	real AngleEvQ, S2evQ, G2evQ, F2evQ, G2evR, F2evR;
	real Gamma2evR, Gamma2fevR, C3Gamma3, C3Gamma3f, G3K, F3K, AngleEvR;
    real A, ApOverf0, CFD3, CFD3p;


    if (_KERNELS_LCDMfk_==1) {
		A=KA_LCDM; ApOverf0 = KAp_LCDM/gd.f0;
		CFD3 = KR1_LCDM;  CFD3p = KR1p_LCDM;
	} else {
		A=1; ApOverf0 = 0;
		CFD3 = 1;  CFD3p = 1;
	}

	real P22dd_p =0.0, P22dd_A = 0.0, P22dd_B = 0.0;
    real P22du_p =0.0, P22du_A = 0.0, P22du_B = 0.0;
    real P22uu_p =0.0, P22uu_A = 0.0, P22uu_B = 0.0;

    real I1udd1tA_p =0.0, I1udd1tA_A = 0.0, I1udd1tA_B = 0.0;
    real I2uud1tA_p =0.0, I2uud1tA_A = 0.0, I2uud1tA_B = 0.0;
    real I2uud2tA_p =0.0, I2uud2tA_A = 0.0, I2uud2tA_B = 0.0;
    real I3uuu2tA_p =0.0, I3uuu2tA_A = 0.0, I3uuu2tA_B = 0.0;
    real I3uuu3tA_p =0.0, I3uuu3tA_A = 0.0, I3uuu3tA_B = 0.0;

    real I2uudd1BpC_p =0.0, I2uudd1BpC_A =0.0, I2uudd1BpC_B =0.0;
    real I2uudd2BpC_p =0.0, I2uudd2BpC_A =0.0, I2uudd2BpC_B =0.0 ;
    real I3uuud2BpC_p =0.0, I3uuud2BpC_A =0.0, I3uuud2BpC_B =0.0 ;
    real I3uuud3BpC_p =0.0, I3uuud3BpC_A =0.0, I3uuud3BpC_B =0.0 ;
    real I4uuuu2BpC_p =0.0, I4uuuu2BpC_A =0.0, I4uuuu2BpC_B =0.0 ;
    real I4uuuu3BpC_p =0.0, I4uuuu3BpC_A =0.0, I4uuuu3BpC_B =0.0 ;
    real I4uuuu4BpC_p =0.0, I4uuuu4BpC_A =0.0, I4uuuu4BpC_B =0.0 ;

    real  Pb1b2_p=0.0,     Pb1b2_A =0.0,     Pb1b2_B =0.0;
    real  Pb1bs2_p=0.0,    Pb1bs2_A =0.0,    Pb1bs2_B =0.0;
    real  Pb22_p=0.0,      Pb22_A =0.0,      Pb22_B =0.0;
    real  Pb2s2_p=0.0,     Pb2s2_A =0.0,     Pb2s2_B =0.0;
    real  Ps22_p=0.0,      Ps22_A =0.0,      Ps22_B =0.0;
    real  Pb2theta_p=0.0,  Pb2theta_A =0.0,  Pb2theta_B =0.0;
    real  Pbs2theta_p=0.0, Pbs2theta_A =0.0, Pbs2theta_B =0.0;

    real KP22dd, KP22du, KP22uu;
    real KI1udd1tA, KI2uud1tA, KI2uud2tA, KI3uuu2tA, KI3uuu3tA;
    real KI2uudd1BpC, KI2uudd2BpC, KI3uuud2BpC, KI3uuud3BpC;
    real KI4uuuu2BpC, KI4uuuu3BpC, KI4uuuu4BpC;
    real KPb1b2, KPb1bs2, KPb22, KPb2s2, KPs22, KPb2theta, KPbs2theta;

	real P13dd_p =0.0, P13dd_A = 0.0, P13dd_B = 0.0;
    real P13du_p =0.0, P13du_A = 0.0, P13du_B = 0.0;
    real P13uu_p =0.0, P13uu_A = 0.0, P13uu_B = 0.0;

    real sigma32PSL_p =0.0, sigma32PSL_A = 0.0, sigma32PSL_B = 0.0;

    real I1udd1a_p =0.0, I1udd1a_A = 0.0, I1udd1a_B = 0.0;
    real I2uud1a_p =0.0, I2uud1a_A = 0.0, I2uud1a_B = 0.0;
    real I2uud2a_p =0.0, I2uud2a_A = 0.0, I2uud2a_B = 0.0;
    real I3uuu2a_p =0.0, I3uuu2a_A = 0.0, I3uuu2a_B = 0.0;
    real I3uuu3a_p =0.0, I3uuu3a_A = 0.0, I3uuu3a_B = 0.0;

    real KP13dd, KP13du, KP13uu, Ksigma32PSL;
    real KI1udd1a, KI2uud1a, KI2uud2a, KI3uuu2a, KI3uuu3a;

    real *kk, *dkk;
    //
    pointPSTableptr p;
    //
    global_kFs_ptr QRstmp;

    QRstmp = (global_kFs_ptr) allocate(1 * sizeof(global_kFs));



    kmin = kPS[1];
    kmax = kPS[nPSLT];
    pmin = MAX(kmin,0.01*cmd.kmin);
    pmax = MIN(kmax,16.0*cmd.kmax);


    dk = (rlog10(pmax) - rlog10(pmin))/((real)(cmd.nquadSteps - 1));
    kk=dvector(1,cmd.nquadSteps);
    dkk=dvector(1,cmd.nquadSteps);
    kk[1] = rpow(10.0,rlog10(pmin));
    for (i=2; i<cmd.nquadSteps; i++) {
        ypi = rlog10(pmin) + dk*((real)(i - 1));
        kk[i] = rpow(10.0,ypi);
        dkk[i] = (kk[i]-kk[i-1]);
    }

// Q functions

    t_q_start = second();
    PSLA = 0.0;
    rmax = kmax/ki;
    rmin = kmin/ki;
    Nx=10;
    xxGL=dvector(1,Nx);
    wwGL=dvector(1,Nx);

	for (i=2; i<cmd.nquadSteps; i++) {
		r = kk[i]/ki;
        r2= r*r;
		PSLB = Interpolation_nr(kk[i], kPKL, pPKL, nPKLT, pPKL2);

		pklp=PSLB;
		fp = Interpolation_nr(kk[i], kPS, fkT, nPSLT, fkT2);
		fp /= gd.f0;
		mumin = MAX( -1.0, (1.0 + rsqr(r) - rsqr(rmax)) / (2.0*r)  );
		mumax = MIN(   1.0, (1.0  + rsqr(r) - rsqr(rmin)) / (2.0*r)  );

		if (r>=0.5)
			mumax = 0.5/r;
			gauleg(mumin,mumax,xxGL,wwGL,Nx);
        for (j=1; j<=Nx; j++) {
            x = xxGL[j];
            w = wwGL[j];
            x2= x*x;

            y2=1.0 + r2 - 2.0 * r * x;
            y = rsqrt(y2);
            psl = Interpolation_nr(ki * y, kPKL, pPKL, nPKLT, pPKL2);
            pklkmp=psl;
			fkmp = Interpolation_nr(ki * y, kPS, fkT, nPSLT, fkT2);
			fkmp /= gd.f0;

			AngleEvQ = (x - r)/y;
			S2evQ = AngleEvQ *AngleEvQ - 1./3.;
			F2evQ = 1./2. + 3./14. * A + (1./2. - 3./14. * A) * AngleEvQ*AngleEvQ +
					AngleEvQ / 2. * (y/r + r/y);
			G2evQ = 3./14. * A * (fp + fkmp) + 3./14. * ApOverf0
					+ (1./2. * (fp + fkmp) - 3./14. *  A *  (fp + fkmp) -
					3./14. * ApOverf0)*AngleEvQ*AngleEvQ
					+ AngleEvQ/2. *  (fkmp * y/r + fp * r/y);


			KP22dd = 2*r2*F2evQ*F2evQ;
			KP22du = 2*r2*F2evQ*G2evQ;
			KP22uu = 2*r2*G2evQ*G2evQ;
// A


			KI1udd1tA = 2.* (fp * r * x + fkmp * r2 * (1. - r * x)/y2 ) * F2evQ ;

			KI2uud1tA = - fp*fkmp * r2 *(1. - x2)/y2 * F2evQ;

			KI2uud2tA = 2.* (fp* r * x + fkmp *  r2*(1. - r * x)/y2) * G2evQ +
						fp * fkmp * ( r2*(1. - 3.* x2) + 2.* r * x )/y2 * F2evQ;

			KI3uuu2tA = fp * fkmp *  r2 * (x2 - 1.)/y2 * G2evQ ;

			KI3uuu3tA = fp * fkmp * ( r2*(1. - 3.* x2) + 2.* r * x )/y2 * G2evQ;



// B+C
			//~ KI2uudd1BpC =  fp*( fp * (1.-x2) + fkmp * r2 * (-1. + x2) / y2 ) / 2.;
			KI2uudd1BpC = 1/4. * (1.-x2)*(fp*fp + fkmp*fkmp*r2*r2/y2/y2)
						+ fkmp*fp *r2 *(-1.+x2)/y2/2.; //Modificacion_Julio1

			KI2uudd2BpC = ( fp*fp*(-1. + 3.*x2) +
				 2 * fkmp * fp*r * (r + 2*x - 3*r*x2) / y2 +
				 fkmp * fkmp * r2*
				  ( 2 - 4*r*x + r2*(-1 + 3*x2) )/(y2*y2)    )/4. ;

			KI3uuud2BpC = -(   fkmp*fp*(fkmp*(-2 + 3*r*x)*r2 -
				 fp*(-1 + 3*r*x)*(1 - 2*r*x + r2))*(-1 + x2 )    )/ (2.* y2*y2)  ;

  			KI3uuud3BpC =   (  fkmp*fp*( -(fp*(1 - 2*r*x + r2)*(1 - 3*x2 + r*x*(-3 + 5*x2))) +
				   fkmp*r*(2*x + r*(2 - 6*x2 + r*x*(-3 + 5*x2))) ) ) / (2.* y2*y2)  ;

			KI4uuuu2BpC =  (3*rpow(fkmp,2)*rpow(fp,2)*r2*rpow(-1 + x2,2)) / (16. *y2*y2 )  ;

			KI4uuuu3BpC = -(rpow(fkmp,2)*rpow(fp,2)*(-1 + x2)*
				  (2 + 3*r*(-4*x + r*(-1 + 5*x2))))  / (8. *y2*y2 );

			KI4uuuu4BpC =  (rpow(fkmp,2)*rpow(fp,2)*(-4 + 8*r*x*(3 - 5*x2) +
				   12*x2 + r2*(3 - 30*x2 + 35*rpow(x,4))))  / (16. *y2*y2 );


  // Bias

			KPb1b2  = r2 * F2evQ;
			KPb1bs2 = r2 * F2evQ*S2evQ;
			KPb22   = 1./2. * r2 * (1./2. * (1. - pklp/pklkmp)
							+ 1./2. * (1. - pklkmp/pklp));
			KPb2s2  = 1./2. * r2 * (  1./2.* (S2evQ - 2./3. * pklp/pklkmp)
							        + 1./2.* (S2evQ - 2./3. * pklkmp/pklp));
			KPs22   = 1./2. * r2 * (1./2. * (S2evQ * S2evQ - 4./9. * pklp/pklkmp)
							+ 1./2.* (S2evQ * S2evQ - 4./9. * pklkmp/pklp));
			KPb2theta  = r2 * G2evQ;
			KPbs2theta = r2 * S2evQ * G2evQ;

            //

            P22dd_B +=   w*KP22dd*psl;
            P22du_B +=   w*KP22du*psl;
            P22uu_B +=   w*KP22uu*psl;

            I1udd1tA_B +=   w*KI1udd1tA*psl;
            I2uud1tA_B +=   w*KI2uud1tA*psl;
            I2uud2tA_B +=   w*KI2uud2tA*psl;
            I3uuu2tA_B +=   w*KI3uuu2tA*psl;
            I3uuu3tA_B +=   w*KI3uuu3tA*psl;

            I2uudd1BpC_B +=   w*KI2uudd1BpC*psl;
            I2uudd2BpC_B +=   w*KI2uudd2BpC*psl;
            I3uuud2BpC_B +=   w*KI3uuud2BpC*psl;
            I3uuud3BpC_B +=   w*KI3uuud3BpC*psl;
            I4uuuu2BpC_B +=   w*KI4uuuu2BpC*psl;
            I4uuuu3BpC_B +=   w*KI4uuuu3BpC*psl;
            I4uuuu4BpC_B +=   w*KI4uuuu4BpC*psl;


            Pb1b2_B     +=   w*KPb1b2*psl;
            Pb1bs2_B    +=   w*KPb1bs2*psl;
            Pb22_B      +=   w*KPb22*psl;
            Pb2s2_B     +=   w*KPb2s2*psl;
            Ps22_B      +=   w*KPs22*psl;
            Pb2theta_B  +=   w*KPb2theta*psl;
            Pbs2theta_B +=   w*KPbs2theta*psl;

        }

        P22dd_p   += dkk[i]*(P22dd_A*PSLA + P22dd_B*PSLB)/2.0;
        P22du_p   += dkk[i]*(P22du_A*PSLA + P22du_B*PSLB)/2.0;
        P22uu_p   += dkk[i]*(P22uu_A*PSLA + P22uu_B*PSLB)/2.0;

        I1udd1tA_p   += dkk[i]*(I1udd1tA_A*PSLA + I1udd1tA_B*PSLB)/2.0;
        I2uud1tA_p   += dkk[i]*(I2uud1tA_A*PSLA + I2uud1tA_B*PSLB)/2.0;
        I2uud2tA_p   += dkk[i]*(I2uud2tA_A*PSLA + I2uud2tA_B*PSLB)/2.0;
        I3uuu2tA_p   += dkk[i]*(I3uuu2tA_A*PSLA + I3uuu2tA_B*PSLB)/2.0;
        I3uuu3tA_p   += dkk[i]*(I3uuu3tA_A*PSLA + I3uuu3tA_B*PSLB)/2.0;

        I2uudd1BpC_p   += dkk[i]*(I2uudd1BpC_A*PSLA + I2uudd1BpC_B*PSLB)/2.0;
        I2uudd2BpC_p   += dkk[i]*(I2uudd2BpC_A*PSLA + I2uudd2BpC_B*PSLB)/2.0;
        I3uuud2BpC_p   += dkk[i]*(I3uuud2BpC_A*PSLA + I3uuud2BpC_B*PSLB)/2.0;
        I3uuud3BpC_p   += dkk[i]*(I3uuud3BpC_A*PSLA + I3uuud3BpC_B*PSLB)/2.0;
        I4uuuu2BpC_p   += dkk[i]*(I4uuuu2BpC_A*PSLA + I4uuuu2BpC_B*PSLB)/2.0;
        I4uuuu3BpC_p   += dkk[i]*(I4uuuu3BpC_A*PSLA + I4uuuu3BpC_B*PSLB)/2.0;
        I4uuuu4BpC_p   += dkk[i]*(I4uuuu4BpC_A*PSLA + I4uuuu4BpC_B*PSLB)/2.0;

        Pb1b2_p       += dkk[i]*(Pb1b2_A    *PSLA + Pb1b2_B    *PSLB)/2.0;
        Pb1bs2_p      += dkk[i]*(Pb1bs2_A   *PSLA + Pb1bs2_B   *PSLB)/2.0;
        Pb22_p        += dkk[i]*(Pb22_A     *PSLA + Pb22_B     *PSLB)/2.0;
        Pb2s2_p       += dkk[i]*(Pb2s2_A    *PSLA + Pb2s2_B    *PSLB)/2.0;
        Ps22_p        += dkk[i]*(Ps22_A     *PSLA + Ps22_B     *PSLB)/2.0;
        Pb2theta_p    += dkk[i]*(Pb2theta_A *PSLA + Pb2theta_B *PSLB)/2.0;
        Pbs2theta_p   += dkk[i]*(Pbs2theta_A*PSLA + Pbs2theta_B*PSLB)/2.0;


        P22dd_A =   P22dd_B;      P22dd_B = 0.0;
        P22du_A =   P22du_B;      P22du_B = 0.0;
        P22uu_A =   P22uu_B;      P22uu_B = 0.0;

        I1udd1tA_A   =   I1udd1tA_B;        I1udd1tA_B   = 0.0;
        I2uud1tA_A   =   I2uud1tA_B;        I2uud1tA_B   = 0.0;
        I2uud2tA_A   =   I2uud2tA_B;        I2uud2tA_B   = 0.0;
        I3uuu2tA_A   =   I3uuu2tA_B;        I3uuu2tA_B   = 0.0;
        I3uuu3tA_A   =   I3uuu3tA_B;        I3uuu3tA_B   = 0.0;

        I2uudd1BpC_A =   I2uudd1BpC_B;      I2uudd1BpC_B = 0.0;
        I2uudd2BpC_A =   I2uudd2BpC_B;      I2uudd2BpC_B = 0.0;
        I3uuud2BpC_A =   I3uuud2BpC_B;      I3uuud2BpC_B = 0.0;
        I3uuud3BpC_A =   I3uuud3BpC_B;      I3uuud3BpC_B = 0.0;
        I4uuuu2BpC_A =   I4uuuu2BpC_B;      I4uuuu2BpC_B = 0.0;
        I4uuuu3BpC_A =   I4uuuu3BpC_B;      I4uuuu3BpC_B = 0.0;
        I4uuuu4BpC_A =   I4uuuu4BpC_B;      I4uuuu4BpC_B = 0.0;

        Pb1b2_A     =   Pb1b2_B;        Pb1b2_B     = 0.0;
        Pb1bs2_A    =   Pb1bs2_B;       Pb1bs2_B    = 0.0;
        Pb22_A      =   Pb22_B;         Pb22_B      = 0.0;
        Pb2s2_A     =   Pb2s2_B;        Pb2s2_B     = 0.0;
        Ps22_A      =   Ps22_B;         Ps22_B      = 0.0;
        Pb2theta_A  =   Pb2theta_B;     Pb2theta_B  = 0.0;
        Pbs2theta_A =   Pbs2theta_B;    Pbs2theta_B = 0.0;




        PSLA = PSLB;
    }

	P22dd_p   *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	P22du_p   *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	P22uu_p   *= 2.0*(rpow(ki,3)/FOURPI2)/ki;

	I1udd1tA_p   *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I2uud1tA_p   *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I2uud2tA_p   *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I3uuu2tA_p   *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I3uuu3tA_p   *= 2.0*(rpow(ki,3)/FOURPI2)/ki;

    I2uudd1BpC_p    *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I2uudd2BpC_p    *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I3uuud2BpC_p    *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I3uuud3BpC_p    *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I4uuuu2BpC_p    *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I4uuuu3BpC_p    *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	I4uuuu4BpC_p    *= 2.0*(rpow(ki,3)/FOURPI2)/ki;

	Pb1b2_p        *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	Pb1bs2_p       *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	Pb22_p         *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	Pb2s2_p        *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	Ps22_p         *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	Pb2theta_p     *= 2.0*(rpow(ki,3)/FOURPI2)/ki;
	Pbs2theta_p    *= 2.0*(rpow(ki,3)/FOURPI2)/ki;

    free_dvector(wwGL,1,Nx);
    free_dvector(xxGL,1,Nx);

    t_q_loop = second() - t_q_start;

//  R functions

    t_q_start = second();  // reuse variable for R-loop start time

    nGL=10;
    xGL=dvector(1,nGL);
    wGL=dvector(1,nGL);
    gauleg(-1.0,1.0,xGL,wGL,nGL);

    //~ fk = 1.0; // This is f(k)/f0 we need to interpolate f(k)
	fk = Interpolation_nr(ki, kPS, fkT, nPSLT, fkT2);
	fk /= gd.f0;
    for (i=2; i<cmd.nquadSteps; i++) {
        r = kk[i]/ki;
        r2= r*r;
        //~ psl = psInterpolation_nr(kk[i], kPS, pPS, nPSLT);
        psl = Interpolation_nr(kk[i], kPKL, pPKL, nPKLT, pPKL2); ;
        fp = Interpolation_nr(kk[i], kPS, fkT, nPSLT, fkT2);
		fp /= gd.f0;
        for (j=1; j<=nGL; j++) {
            x = xGL[j];
            w = wGL[j];
            x2 =x*x;
            y2=1.0 + r2 - 2.0 * r * x;


			Gamma2evR  = A *(1. - x2);
			Gamma2fevR = A *(1. - x2)*(fk + fp)/2. + 1./2. * ApOverf0 *(1 - x2);

			C3Gamma3  = 2.*5./21. * CFD3  *(1 - x2)*(1 - x2)/y2;
			C3Gamma3f = 2.*5./21. * CFD3p *(1 - x2)*(1 - x2)/y2 *(fk + 2 * fp)/3.;

			G3K = C3Gamma3f/ 2. + (2 * Gamma2fevR * x)/(7. * r) - (fk  * x2)/(6 * r2)
				+ fp * Gamma2evR*(1 - r * x)/(7 * y2)
				- 1./7.*(fp * Gamma2evR + 2 *Gamma2fevR) * (1. - x2)/y2;
			F3K = C3Gamma3/6. - x2/(6 * r2) + (Gamma2evR * x *(1 - r * x))/(7. *r *y2);

			AngleEvR = -x;
			F2evR = 1./2. + 3./14. * A + (1./2. - 3./14. * A) * AngleEvR*AngleEvR
				+ AngleEvR/2. *(1./r + r);

			G2evR = 3./14.* A *(fp + fk) + 3./14.* ApOverf0
				+ ((fp + fk)/2. - 3./14.* A *(fp + fk) - 3./14.* ApOverf0)*AngleEvR*AngleEvR
				+ AngleEvR/2. * (fk/r + fp * r);

			KP13dd = 6.* r2 * F3K;
			KP13du = 3.* r2 * G3K + 3.* r2 * F3K * fk;
			KP13uu = 6.* r2 * G3K * fk;

			Ksigma32PSL = ( 5.0* r2 * (7. - 2*r2 + 4*r*x + 6*(-2 + r2)*x2 -
				12*r*x2*x + 9*x2*x2)) / (24.0 * y2 ) ;

			KI1udd1a = 2.*r2*(1 - r * x)/y2 *G2evR + 2*fp *r* x* F2evR ;

			KI2uud1a = -fp * r2 * (1 - x2)/y2*G2evR ;

			KI2uud2a = ( (r2 *(1 - 3.*x2) + 2.* r* x) /y2*fp +
				  fk*2.*r2*(1 - r * x)/y2)* G2evR + 2*x*r*fp*fk*F2evR;


			KI3uuu2a = fk * KI2uud1a;

			KI3uuu3a = (r2 *(1 - 3.*x2) + 2.* r* x)/y2 * fp*fk * G2evR;


            P13dd_B +=   w*KP13dd*psl;
            P13du_B +=   w*KP13du*psl;
            P13uu_B +=   w*KP13uu*psl;

            sigma32PSL_B +=   w*Ksigma32PSL*psl;

            I1udd1a_B +=   w*KI1udd1a*psl;
            I2uud1a_B +=   w*KI2uud1a*psl;
            I2uud2a_B +=   w*KI2uud2a*psl;
            I3uuu2a_B +=   w*KI3uuu2a*psl;
            I3uuu3a_B +=   w*KI3uuu3a*psl;

        }



        P13dd_p   += dkk[i]*(P13dd_A + P13dd_B) /  (2.0*ki);
        P13du_p   += dkk[i]*(P13du_A + P13du_B) /  (2.0*ki);
        P13uu_p   += dkk[i]*(P13uu_A + P13uu_B) /  (2.0*ki);

        sigma32PSL_p   += dkk[i]*(sigma32PSL_A + sigma32PSL_B) /  (2.0*ki);

        I1udd1a_p   += dkk[i]*(I1udd1a_A + I1udd1a_B) /  (2.0*ki);
        I2uud1a_p   += dkk[i]*(I2uud1a_A + I2uud1a_B) /  (2.0*ki);
        I2uud2a_p   += dkk[i]*(I2uud2a_A + I2uud2a_B) /  (2.0*ki);
        I3uuu2a_p   += dkk[i]*(I3uuu2a_A + I3uuu2a_B) /  (2.0*ki);
        I3uuu3a_p   += dkk[i]*(I3uuu3a_A + I3uuu3a_B) /  (2.0*ki);


        P13dd_A =   P13dd_B;      P13dd_B = 0.0;
        P13du_A =   P13du_B;      P13du_B = 0.0;
        P13uu_A =   P13uu_B;      P13uu_B = 0.0;
        sigma32PSL_A =   sigma32PSL_B;      sigma32PSL_B = 0.0;

        I1udd1a_A =   I1udd1a_B;      I1udd1a_B = 0.0;
        I2uud1a_A =   I2uud1a_B;      I2uud1a_B = 0.0;
        I2uud2a_A =   I2uud2a_B;      I2uud2a_B = 0.0;
        I3uuu2a_A =   I3uuu2a_B;      I3uuu2a_B = 0.0;
        I3uuu3a_A =   I3uuu3a_B;      I3uuu3a_B = 0.0;

    }

    pkl_k = Interpolation_nr(ki, kPKL, pPKL, nPKLT, pPKL2);
    P13dd_p      *= (rpow(ki,3.0)/FOURPI2)*pkl_k;
    P13du_p      *= (rpow(ki,3.0)/FOURPI2)*pkl_k;
    P13uu_p      *= (rpow(ki,3.0)/FOURPI2)*pkl_k;
    sigma32PSL_p *= (rpow(ki,3.0)/FOURPI2)*pkl_k;
    I1udd1a_p    *= (rpow(ki,3.0)/FOURPI2)*pkl_k;
    I2uud1a_p    *= (rpow(ki,3.0)/FOURPI2)*pkl_k;
    I2uud2a_p    *= (rpow(ki,3.0)/FOURPI2)*pkl_k;
    I3uuu2a_p    *= (rpow(ki,3.0)/FOURPI2)*pkl_k;
    I3uuu3a_p    *= (rpow(ki,3.0)/FOURPI2)*pkl_k;



	kFs(QRstmp)    = ki;

	P22dd(  QRstmp)      = P22dd_p;
	P22du(  QRstmp)      = P22du_p;
	P22uu(  QRstmp)      = P22uu_p;
	// A TNS
	I1udd1A(  QRstmp)      = I1udd1tA_p + 2.0*I1udd1a_p;
	I2uud1A(  QRstmp)      = I2uud1tA_p + 2.0*I2uud1a_p;
	I2uud2A(  QRstmp)      = I2uud2tA_p + 2.0*I2uud2a_p;
	I3uuu2A(  QRstmp)      = I3uuu2tA_p + 2.0*I3uuu2a_p;
	I3uuu3A(  QRstmp)      = I3uuu3tA_p + 2.0*I3uuu3a_p;
	// D function: B + C - G
	I2uudd1BpC(  QRstmp)   = I2uudd1BpC_p
								- ki*ki*gd.sigma2v*pkl_k;
	//~ I2uudd1BpC(  QRstmp)   = I2uudd1BpC_p;
	I2uudd2BpC(  QRstmp)   = I2uudd2BpC_p;
	I3uuud2BpC(  QRstmp)   = I3uuud2BpC_p
								- 2.0*ki*ki*gd.sigma2v*fk*pkl_k;
	//~ I3uuud2BpC(  QRstmp)   = I3uuud2BpC_p;
	I3uuud3BpC(  QRstmp)   = I3uuud3BpC_p;
	I4uuuu2BpC(  QRstmp)   = I4uuuu2BpC_p;
	I4uuuu3BpC(  QRstmp)   = I4uuuu3BpC_p
								- ki*ki*gd.sigma2v*fk*fk*pkl_k;
	//~ I4uuuu3BpC(  QRstmp)   = I4uuuu3BpC_p;
	I4uuuu4BpC(  QRstmp)   = I4uuuu4BpC_p;



	//  Bias
	Pb1b2(    QRstmp) = Pb1b2_p;
	Pb1bs2(   QRstmp) = Pb1bs2_p;
	Pb22(     QRstmp) = Pb22_p;
	Pb2s2(    QRstmp) = Pb2s2_p;
	Ps22(     QRstmp) = Ps22_p;
	Pb2theta( QRstmp) = Pb2theta_p;
	Pbs2theta(QRstmp) = Pbs2theta_p;
	//
	P13dd(  QRstmp)      = P13dd_p;
	P13du(  QRstmp)      = P13du_p;
	P13uu(  QRstmp)      = P13uu_p;

	sigma32PSL(QRstmp)   = sigma32PSL_p;


    free_dvector(dkk,1,cmd.nquadSteps);
    free_dvector(kk,1,cmd.nquadSteps);

    t_r_loop = second() - t_q_start;

    // Optional detailed timing output (controlled by chatty==2 for extra verbosity)
    if(cmd.chatty >= 2) {
        fprintf(stdout,"\n    ki=%9.6f: Q-loop=%5.2fms, R-loop=%5.2fms",
                ki, 1e3*t_q_loop, 1e3*t_r_loop);
    }

    return *QRstmp;
}









































// END Qs and Rs

//~ global_kFs qrs;
global_kFs kfunctions;
//~ global_kFs qrs_nw;

//~ global global_kFs ki_functions_driver(real eta, real ki)
//~ global global_kFs ki_functions_driver(real ki)
global global_kFs ki_functions_driver(real ki, double kPKL[], double pPKL[], int nPKLT, double pPKL2[])
{
    //~ quadrature(ki);
    kfunctions = ki_functions(ki,kPKL,pPKL,nPKLT,pPKL2);
    //~ return qrs;
    return kfunctions;
}


#define ROMO 1
#define NULLMETHOD 0
#define TRAPEZOID 2
#define TRAPEZOID3 5





void quadraturemethod_string_to_int(string method_str,int *method_int)
{
    *method_int=-1;
    if (strcmp(method_str,"romberg") == 0) {
        *method_int = ROMO;
        strcpy(gd.quadraturemethod_comment, "romberg open quadrature method");
    }
//
    if (strcmp(method_str,"trapezoid") == 0) {
        *method_int = TRAPEZOID;
        strcpy(gd.quadraturemethod_comment, "trapezoid quadrature method");
    }
//
    if (strcmp(method_str,"trapezoid3") == 0) {
        *method_int = TRAPEZOID3;
        strcpy(gd.quadraturemethod_comment, "trapezoid3 quadrature method");
    }
//
    if (strnull(method_str)) {
        *method_int = NULLMETHOD;
        strcpy(gd.quadraturemethod_comment,
               "given null quadrature method ... running deafult (trapezoid)");
        fprintf(stdout,"\n\tintegration: default integration method (trapezoid)...\n");
    }
//
    if (*method_int == -1) {
        *method_int = TRAPEZOID;
        strcpy(gd.quadraturemethod_comment,
               "Unknown quadrature method ... running deafult (trapezoid)");
        fprintf(stdout,"\n\tquadrature: Unknown method... %s ",cmd.quadratureMethod);
        fprintf(stdout,
                "\n\trunning default quadrature method (trapezoid)...\n");
    }
}

#undef ROMO
#undef TRAPEZOID
#undef TRAPEZOID3
#undef NULLMETHOD



#undef KK
#undef QROMBERG




local  real Interpolation_nr(real k, double kPS[], double pPS[], int nPS, double pPS2[])
{
    real psftmp;
    splint(kPS,pPS,pPS2,nPS,k,&psftmp);
    return (psftmp);
}


local real sigma28_function_int(real y)
{
    real p;
    real PSL,fk,j1, W;

    p = rpow(10.0,y);
    PSL = psInterpolation_nr(p, kPS, pPS, nPSLT);
    j1 = rj1Bessel(p*8.0);
    W = 3.0*j1/(p*8.0);

    return p*p*p*PSL*W*W;
}


local real sigma2L_function_int(real y)
{
    real p;
    real PSL;

    p = rpow(10.0,y);
    PSL = psInterpolation_nr(p, kPS, pPS, nPSLT);

    return p*PSL;
}


local real sigma2v_function_int(real y)
{
    real p;
    real PSL,fk;

    p = rpow(10.0,y);
    PSL = psInterpolation_nr(p, kPS, pPS, nPSLT);
    fk = Interpolation_nr(p, kPS, fkT, nPSLT, fkT2);
	fk /= gd.f0;
    return p*PSL*fk*fk;
}

local real Sigma2_int(real y)
{
    real p, PSL_nw, kosc;

    kosc=1.0/104.;


    p = rpow(10.0,y);
    PSL_nw = Interpolation_nr(p, kPS, pPS_nw, nPSLT,pPS2_nw);
    return p * PSL_nw * (1- rj0Bessel(p/kosc) + 2.*rj2Bessel(p/kosc) );
}



local real deltaSigma2_int(real y)
{
    real p, PSL_nw, kosc;

    kosc=1.0/104.;


    p = rpow(10.0,y);
    PSL_nw = Interpolation_nr(p, kPS, pPS_nw, nPSLT,pPS2_nw);
    return 3.0 * p * PSL_nw * rj2Bessel(p/kosc) ;
}




local real sigma_constants(void)
{

    real sigma2v, sigma2L, Sigma2, deltaSigma2;
    real kmin, kmax;
    real ymin, ymax, ymaxSigma;
    real EPSQ = 0.000001;
    int KK = 5;
	real ks;


    kmin = kPS[1];
    kmax = kPS[nPSLT];
    ymin = rlog10(kmin);
    ymax = rlog10(kmax);

    sigma2v= (1.0/SIXPI2)*rlog(10.0)*
				qromo(sigma2v_function_int,ymin,ymax,midpnt,EPSQ,KK);
    gd.sigma2v = sigma2v;


    sigma2L= (1.0/SIXPI2)*rlog(10.0)*
				qromo(sigma2L_function_int,ymin,ymax,midpnt,EPSQ,KK);
    gd.sigma2L = sigma2L;


	ks = 0.4;
    ymaxSigma = rlog10(ks);

    Sigma2 = (1.0/SIXPI2)*rlog(10.0)*
				qromo(Sigma2_int,ymin,ymaxSigma,midpnt,EPSQ,KK);
    gd.Sigma2 = Sigma2;

    deltaSigma2 = (1.0/SIXPI2)*rlog(10.0)*
				qromo(deltaSigma2_int,ymin,ymaxSigma,midpnt,EPSQ,KK);
    gd.deltaSigma2 = deltaSigma2;

};



local real get_sigma8(void)
{

    real sigma28, sigma8;
    real kmin, kmax;
    real ymin, ymax, ymaxSigma;
    real EPSQ = 0.000001;
    int KK = 5;
	real ks;


    kmin = kPS[1];
    //~ kmax = kPS[nPSLT];
    kmax = 1.0;
    ymin = rlog10(kmin);
    ymax = rlog10(kmax);

	sigma28 = 3*(1.0/SIXPI2)*rlog(10.0)*
				qromo(sigma28_function_int,ymin,ymaxSigma,midpnt,EPSQ,KK);
	gd.sigma8 = rsqrt(sigma28);
};


// HDF5 dump implementation
#ifdef USE_HDF5
#include <hdf5.h>
#include <time.h>

local void dump_kfunctions_hdf5(const char *filename)
{
    hid_t file_id, group_id, dataset_id, dataspace_id, attr_id, attrspace_id;
    hsize_t dims[2];
    herr_t status;
    int i;
    time_t current_time;
    char time_str[100];

    // Create HDF5 file
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error: Could not create HDF5 file %s\n", filename);
        return;
    }

    // Get current timestamp
    time(&current_time);
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&current_time));

    // Create metadata group
    group_id = H5Gcreate(file_id, "/metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    attrspace_id = H5Screate(H5S_SCALAR);

    // Store timestamp
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, strlen(time_str) + 1);
    attr_id = H5Acreate(group_id, "timestamp", str_type, attrspace_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, str_type, time_str);
    H5Aclose(attr_id);

    // Store command line info (reconstructed from parameters)
    char cmdline[512];
    snprintf(cmdline, sizeof(cmdline), "Om=%.4f h=%.4f model=%s fR0=%.3e zout=%.2f fnamePS=%s",
             cmd.om, cmd.h, cmd.mgmodel, cmd.fR0, cmd.xstop, cmd.fnamePS);
    H5Tset_size(str_type, strlen(cmdline) + 1);
    attr_id = H5Acreate(group_id, "command_line", str_type, attrspace_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, str_type, cmdline);
    H5Aclose(attr_id);
    H5Tclose(str_type);

    H5Sclose(attrspace_id);
    H5Gclose(group_id);

    // Create parameters group
    group_id = H5Gcreate(file_id, "/parameters", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Cosmology subgroup
    hid_t cosmo_group = H5Gcreate(group_id, "cosmology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims[0] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);

    #define WRITE_SCALAR(name, value) \
        dataset_id = H5Dcreate(cosmo_group, name, H5T_NATIVE_DOUBLE, dataspace_id, \
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value); \
        H5Dclose(dataset_id);

    WRITE_SCALAR("Om", cmd.om);
    WRITE_SCALAR("h", cmd.h);
    WRITE_SCALAR("zout", cmd.xstop);
    WRITE_SCALAR("f0", gd.f0);
    WRITE_SCALAR("Dplus", gd.Dplus);

    #undef WRITE_SCALAR
    H5Sclose(dataspace_id);
    H5Gclose(cosmo_group);

    // Model subgroup
    hid_t model_group = H5Gcreate(group_id, "model", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, strlen(cmd.mgmodel) + 1);
    dims[0] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(model_group, "mgmodel", str_type, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &cmd.mgmodel);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Tclose(str_type);

    dims[0] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(model_group, "fR0", H5T_NATIVE_DOUBLE, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &cmd.fR0);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Gclose(model_group);

    // k_grid subgroup
    hid_t kgrid_group = H5Gcreate(group_id, "k_grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims[0] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);

    #define WRITE_SCALAR_KGRID(name, value) \
        dataset_id = H5Dcreate(kgrid_group, name, H5T_NATIVE_DOUBLE, dataspace_id, \
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value); \
        H5Dclose(dataset_id);

    WRITE_SCALAR_KGRID("kmin", cmd.kmin);
    WRITE_SCALAR_KGRID("kmax", cmd.kmax);

    #undef WRITE_SCALAR_KGRID

    dataset_id = H5Dcreate(kgrid_group, "Nk", H5T_NATIVE_INT, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &cmd.Nk);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Gclose(kgrid_group);

    // Numerical parameters subgroup
    hid_t numerical_group = H5Gcreate(group_id, "numerical", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims[0] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(numerical_group, "nquadSteps", H5T_NATIVE_INT, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &cmd.nquadSteps);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Gclose(numerical_group);

    // Kernels subgroup
    hid_t kernels_group = H5Gcreate(group_id, "kernels", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims[0] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);

    #define WRITE_KERNEL(name, value) \
        dataset_id = H5Dcreate(kernels_group, name, H5T_NATIVE_DOUBLE, dataspace_id, \
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value); \
        H5Dclose(dataset_id);

    WRITE_KERNEL("KA_LCDM", KA_LCDM);
    WRITE_KERNEL("KAp_LCDM", KAp_LCDM);
    WRITE_KERNEL("KB_LCDM", KB_LCDM);
    WRITE_KERNEL("KR1_LCDM", KR1_LCDM);
    WRITE_KERNEL("KR1p_LCDM", KR1p_LCDM);

    #undef WRITE_KERNEL
    H5Sclose(dataspace_id);
    H5Gclose(kernels_group);
    H5Gclose(group_id);

    // Create sigma_values group
    group_id = H5Gcreate(file_id, "/sigma_values", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims[0] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);

    #define WRITE_SIGMA(name, value) \
        dataset_id = H5Dcreate(group_id, name, H5T_NATIVE_DOUBLE, dataspace_id, \
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value); \
        H5Dclose(dataset_id);

    WRITE_SIGMA("sigma8", gd.sigma8);
    WRITE_SIGMA("sigma2L", gd.sigma2L);
    WRITE_SIGMA("sigma2v", gd.sigma2v);
    WRITE_SIGMA("Sigma2", gd.Sigma2);
    WRITE_SIGMA("deltaSigma2", gd.deltaSigma2);

    #undef WRITE_SIGMA
    H5Sclose(dataspace_id);
    H5Gclose(group_id);

    // Create inputs group - store linear power spectra
    group_id = H5Gcreate(file_id, "/inputs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Wiggle power spectrum: k, P(k), P''(k), f(k)
    dims[0] = nPSLT;
    dims[1] = 4;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(group_id, "linear_ps_wiggle", H5T_NATIVE_DOUBLE, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    double *ps_wiggle = (double *)malloc(nPSLT * 4 * sizeof(double));
    for (i = 0; i < nPSLT; i++) {
        ps_wiggle[i*4 + 0] = kPS[i+1];
        ps_wiggle[i*4 + 1] = pPS[i+1];
        ps_wiggle[i*4 + 2] = pPS2[i+1];
        ps_wiggle[i*4 + 3] = fkT[i+1];
    }
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ps_wiggle);
    free(ps_wiggle);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // No-wiggle power spectrum
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(group_id, "linear_ps_nowiggle", H5T_NATIVE_DOUBLE, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    double *ps_nowiggle = (double *)malloc(nPSLT * 4 * sizeof(double));
    for (i = 0; i < nPSLT; i++) {
        ps_nowiggle[i*4 + 0] = kPS[i+1];
        ps_nowiggle[i*4 + 1] = pPS_nw[i+1];
        ps_nowiggle[i*4 + 2] = pPS2_nw[i+1];
        ps_nowiggle[i*4 + 3] = fkT[i+1];
    }
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ps_nowiggle);
    free(ps_nowiggle);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Gclose(group_id);

    // Create outputs group
    group_id = H5Gcreate(file_id, "/outputs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Wiggle k-functions
    hid_t kfunc_wiggle = H5Gcreate(group_id, "kfunctions_wiggle", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims[0] = cmd.Nk;
    dataspace_id = H5Screate_simple(1, dims, NULL);

    #define WRITE_ARRAY(name, array) \
        dataset_id = H5Dcreate(kfunc_wiggle, name, H5T_NATIVE_DOUBLE, dataspace_id, \
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, array); \
        H5Dclose(dataset_id);

    WRITE_ARRAY("k", kFArrays.kT);
    WRITE_ARRAY("P22dd", kFArrays.P22ddT);
    WRITE_ARRAY("P22du", kFArrays.P22duT);
    WRITE_ARRAY("P22uu", kFArrays.P22uuT);
    WRITE_ARRAY("I1udd1A", kFArrays.I1udd1AT);
    WRITE_ARRAY("I2uud1A", kFArrays.I2uud1AT);
    WRITE_ARRAY("I2uud2A", kFArrays.I2uud2AT);
    WRITE_ARRAY("I3uuu2A", kFArrays.I3uuu2AT);
    WRITE_ARRAY("I3uuu3A", kFArrays.I3uuu3AT);
    WRITE_ARRAY("I2uudd1BpC", kFArrays.I2uudd1BpCT);
    WRITE_ARRAY("I2uudd2BpC", kFArrays.I2uudd2BpCT);
    WRITE_ARRAY("I3uuud2BpC", kFArrays.I3uuud2BpCT);
    WRITE_ARRAY("I3uuud3BpC", kFArrays.I3uuud3BpCT);
    WRITE_ARRAY("I4uuuu2BpC", kFArrays.I4uuuu2BpCT);
    WRITE_ARRAY("I4uuuu3BpC", kFArrays.I4uuuu3BpCT);
    WRITE_ARRAY("I4uuuu4BpC", kFArrays.I4uuuu4BpCT);
    WRITE_ARRAY("Pb1b2", kFArrays.Pb1b2T);
    WRITE_ARRAY("Pb1bs2", kFArrays.Pb1bs2T);
    WRITE_ARRAY("Pb22", kFArrays.Pb22T);
    WRITE_ARRAY("Pb2s2", kFArrays.Pb2s2T);
    WRITE_ARRAY("Ps22", kFArrays.Ps22T);
    WRITE_ARRAY("Pb2theta", kFArrays.Pb2thetaT);
    WRITE_ARRAY("Pbs2theta", kFArrays.Pbs2thetaT);
    WRITE_ARRAY("P13dd", kFArrays.P13ddT);
    WRITE_ARRAY("P13du", kFArrays.P13duT);
    WRITE_ARRAY("P13uu", kFArrays.P13uuT);
    WRITE_ARRAY("sigma32PSL", kFArrays.sigma32PSLT);
    WRITE_ARRAY("pkl", kFArrays.pklT);
    WRITE_ARRAY("fk", kFArrays.fkT);

    #undef WRITE_ARRAY
    H5Sclose(dataspace_id);
    H5Gclose(kfunc_wiggle);

    // No-wiggle k-functions
    hid_t kfunc_nowiggle = H5Gcreate(group_id, "kfunctions_nowiggle", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims[0] = cmd.Nk;
    dataspace_id = H5Screate_simple(1, dims, NULL);

    #define WRITE_ARRAY_NW(name, array) \
        dataset_id = H5Dcreate(kfunc_nowiggle, name, H5T_NATIVE_DOUBLE, dataspace_id, \
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, array); \
        H5Dclose(dataset_id);

    WRITE_ARRAY_NW("k", kFArrays_nw.kT);
    WRITE_ARRAY_NW("P22dd", kFArrays_nw.P22ddT);
    WRITE_ARRAY_NW("P22du", kFArrays_nw.P22duT);
    WRITE_ARRAY_NW("P22uu", kFArrays_nw.P22uuT);
    WRITE_ARRAY_NW("I1udd1A", kFArrays_nw.I1udd1AT);
    WRITE_ARRAY_NW("I2uud1A", kFArrays_nw.I2uud1AT);
    WRITE_ARRAY_NW("I2uud2A", kFArrays_nw.I2uud2AT);
    WRITE_ARRAY_NW("I3uuu2A", kFArrays_nw.I3uuu2AT);
    WRITE_ARRAY_NW("I3uuu3A", kFArrays_nw.I3uuu3AT);
    WRITE_ARRAY_NW("I2uudd1BpC", kFArrays_nw.I2uudd1BpCT);
    WRITE_ARRAY_NW("I2uudd2BpC", kFArrays_nw.I2uudd2BpCT);
    WRITE_ARRAY_NW("I3uuud2BpC", kFArrays_nw.I3uuud2BpCT);
    WRITE_ARRAY_NW("I3uuud3BpC", kFArrays_nw.I3uuud3BpCT);
    WRITE_ARRAY_NW("I4uuuu2BpC", kFArrays_nw.I4uuuu2BpCT);
    WRITE_ARRAY_NW("I4uuuu3BpC", kFArrays_nw.I4uuuu3BpCT);
    WRITE_ARRAY_NW("I4uuuu4BpC", kFArrays_nw.I4uuuu4BpCT);
    WRITE_ARRAY_NW("Pb1b2", kFArrays_nw.Pb1b2T);
    WRITE_ARRAY_NW("Pb1bs2", kFArrays_nw.Pb1bs2T);
    WRITE_ARRAY_NW("Pb22", kFArrays_nw.Pb22T);
    WRITE_ARRAY_NW("Pb2s2", kFArrays_nw.Pb2s2T);
    WRITE_ARRAY_NW("Ps22", kFArrays_nw.Ps22T);
    WRITE_ARRAY_NW("Pb2theta", kFArrays_nw.Pb2thetaT);
    WRITE_ARRAY_NW("Pbs2theta", kFArrays_nw.Pbs2thetaT);
    WRITE_ARRAY_NW("P13dd", kFArrays_nw.P13ddT);
    WRITE_ARRAY_NW("P13du", kFArrays_nw.P13duT);
    WRITE_ARRAY_NW("P13uu", kFArrays_nw.P13uuT);
    WRITE_ARRAY_NW("sigma32PSL", kFArrays_nw.sigma32PSLT);
    WRITE_ARRAY_NW("pkl", kFArrays_nw.pklT);
    WRITE_ARRAY_NW("fk", kFArrays_nw.fkT);

    #undef WRITE_ARRAY_NW
    H5Sclose(dataspace_id);
    H5Gclose(kfunc_nowiggle);
    H5Gclose(group_id);

    // Close the file
    H5Fclose(file_id);
}

#else
// Stub function if HDF5 is not available
local void dump_kfunctions_hdf5(const char *filename)
{
    fprintf(stderr, "ERROR: HDF5 support not compiled. Rebuild with -DUSE_HDF5 and link with -lhdf5\n");
    error("HDF5 dump requested but not available\n");
}
#endif





















