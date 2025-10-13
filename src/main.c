/*==============================================================================
 NAME: main.c				[code for redshift space correlation function - GSM]
 Alejandro Aviles (avilescervantes@gmail.com), ...
 *  (other people who collaborated: Mario A. Rodriguez-Meza ...)
 ================================================================================ 
 Use: ./fkpt -help
 References:  arXiv:...
 ==============================================================================*/

#define global

#include "globaldefs.h"
#include "cmdline_defs.h"
#include "protodefs.h"
#include "models.h"

int main(int argc, string argv[])
{
    gd.cpuinit = second();
    InitParam(argv, defv);
    StartRun(argv[0], HEAD1, HEAD2, HEAD3);
    MainLoop();
    EndRun();
    return 0;
}



void MainLoop(void)
{
    real t_start, t_global, t_kfunc, t_rsd, t_write, t_free, t_total;

    t_start = second();
    global_variables();
    t_global = second() - t_start;

    t_start = second();
    compute_kfunctions();
    t_kfunc = second() - t_start;

    t_start = second();
    compute_rsdmultipoles();
    t_rsd = second() - t_start;

    t_start = second();
    write();
    t_write = second() - t_start;

    t_start = second();
    free_variables();
    t_free = second() - t_start;

    t_total = t_global + t_kfunc + t_rsd + t_write + t_free;

    if(cmd.chatty >= 1) {
        fprintf(stdout,"\n\n======================== TIMING BREAKDOWN ========================\n");
        fprintf(stdout,"global_variables:        %8.3f s (%5.1f%%)\n",
                t_global, 100.0*t_global/t_total);
        fprintf(stdout,"compute_kfunctions:      %8.3f s (%5.1f%%)\n",
                t_kfunc, 100.0*t_kfunc/t_total);
        fprintf(stdout,"compute_rsdmultipoles:   %8.3f s (%5.1f%%)\n",
                t_rsd, 100.0*t_rsd/t_total);
        fprintf(stdout,"write:                   %8.3f s (%5.1f%%)\n",
                t_write, 100.0*t_write/t_total);
        fprintf(stdout,"free_variables:          %8.3f s (%5.1f%%)\n",
                t_free, 100.0*t_free/t_total);
        fprintf(stdout,"==================================================================\n");
        fprintf(stdout,"TOTAL MainLoop time:     %8.3f s\n", t_total);
        fprintf(stdout,"==================================================================\n\n");
    }
}
