#ifndef CONTEXT_H_ /* Include guard */
#define CONTEXT_H_

#define DIM 2
#define QUAD_ORDER 3
#define QUAD_PNTS 9
#define CFF_LEN 6
#define LOC_NDS 4
#define LOC_FDMS 8
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern const double QUAD_CORD[QUAD_ORDER];
extern const double QUAD_WGHT[QUAD_ORDER];
extern const double TOL;

extern const double default_T;
extern const double default_Tresca_bnd;
extern const double E0;
extern const double E1;
extern const double nu0;
extern const double nu1;

extern double base_val_at_quad_pnt[LOC_NDS][QUAD_PNTS];
extern double base_grad_val_at_quad_pnt[LOC_NDS][DIM][QUAD_PNTS];
extern double base_grad2_val_at_quad_pnt[LOC_NDS];
extern double quad_wghts[QUAD_PNTS];

void get_locbase_val(unsigned short loc_ind, double x, double y, double *val);

void get_locbase_grad_val(unsigned short loc_ind, double x, double y, double *grad_val);

void get_C_from_E_nu(double E, double nu, double *C);

void default_tr_f1(double x, double t, double *val);
void default_tr_f2(double y, double t, double *val);
void default_bdy_f(double x, double y, double t, double *val);

void default_cell_cff(double x, double y, const void *paras, double *C);
void connected_cell_cff(double x, double y, const void *paras, double *C);
void const_cell_cff(double x, double y, const void *paras, double *C);

typedef void (*cell_cff_func)(double, double, const void *, double *C);
typedef void (*s2d_t1d_to_2d_func)(double, double, double, double *val);
typedef void (*s1d_t1d_to_2d_func)(double, double, double *val);

typedef struct Context
{
    unsigned int prd;
    unsigned int grids_on_cell;
    unsigned int grids_on_dmn;
    unsigned int total_elems;
    unsigned int total_nds;
    unsigned int total_fdms;
    unsigned int total_elems_on_cell;
    unsigned int total_nds_on_cell;
    unsigned int total_fdms_on_cell;
    double hh;
    double h;
    double *cff_data;
} Context;

void Context_init_(Context *ctx, unsigned int prd, unsigned int grids_on_cell, cell_cff_func cell_cff, const void *paras);

void Context_final_(Context *ctx);

#endif
