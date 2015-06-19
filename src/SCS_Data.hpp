#ifndef SCS_DATA_H
#define SCS_DATA_H

#include <vector>
#include <string>
#include "scs.h"
#include "cones.h"
#include "FAO_DAG.hpp"

class SCS_Data {
public:
     double *c;
     size_t c_len;
     double *b;
     size_t b_len;
     /* Solver result info. */
     int statusVal;
     int iter;
     int cgIter;
     double pobj;
     double dobj;
     double resPri;
     double resDual;
     double relGap;
     double solveTime;
     double setupTime;
     char* status;

     double *x;
     double *y;

     ~SCS_Data() {
          free(status);
     }


     void load_c(double *c, int c_len) {
          this->c = c;
          this->c_len = c_len;
     }

     void load_b(double *b, int b_len) {
          this->b = b;
          this->b_len = b_len;
     }

     void load_x(double *x, int x_len) {
          this->x = x;
     }

     void load_y(double *y, int y_len) {
          this->y = y;
     }

     int solve(FAO_DAG* fao_dag,
               int f, int l, std::vector<int> q, std::vector<int> s,
               int ep, size_t max_iters, size_t equil_steps,
               size_t samples, bool precond, double eps,
               bool rand_seed) {
          Cone * k;
          Data * d;
          Info info = { 0 };
          // Work * w;
          Sol * sol;
          k = (Cone *) calloc(1, sizeof(Cone));
          d = (Data *) calloc(1, sizeof(Data));
          sol = (Sol *) calloc(1, sizeof(Sol));

          // Load cones.
          k->f = f;
          k->l = l;
          k->q = q.data();
          k->qsize = q.size();
          k->s = s.data();
          k->ssize = s.size();
          k->ep = ep;
          k->ed = 0;

          // Load c, b, cone info into data structures.
          d->b = b;
          d->c = c;
          d->m = b_len;
          d->n = c_len;
          d->fao_dag = (void *) fao_dag;
          d->Amul = fao_dag->static_forward_eval;
          d->ATmul = fao_dag->static_adjoint_eval;
          d->dag_input = fao_dag->get_forward_input()->data;
          d->dag_output = fao_dag->get_adjoint_input()->data;

          d->CG_RATE = 2.0;
          d->VERBOSE = true;
          d->MAX_ITERS = max_iters;
          d->EPS = eps;
          d->ALPHA = 1.5;
          d->RHO_X = 1e-3;
          d->SCALE = 5;
          d->NORMALIZE = true; /* boolean, heuristic data rescaling: 1 */
          d->WARM_START = false; /* boolean, warm start (put initial guess in Sol struct): 0 */
          d->EQUIL_STEPS = equil_steps; /* how many steps of equilibration to take: 1 */
          d->EQUIL_P = 2;  /* Which Lp norm equilibration used? */
          d->EQUIL_GAMMA = 0; /* Regularization parameter for equilibration: 1e-8 */
          d->STOCH = true; /* Use random approximation of L2 norms? */
          d->SAMPLES = samples; /* Number of samples for approximation: 200 */
          d->PRECOND = precond; /* boolean, use preconditioner for CG? */
          d->RAND_SEED = rand_seed;

          scs(d, k, sol, &info);
          load_info(&info);
          memcpy(this->x, sol->x, sizeof(double)*this->c_len);
          memcpy(this->y, sol->y, sizeof(double)*this->b_len);
          free(d);
          free(k);
          freeSol(sol);
          return 0;
     }

     void load_info(Info *info) {
          this->statusVal = info->statusVal;
          this->iter = info->iter;
          this->cgIter = info->cgIter;
          this->pobj = info->pobj;
          this->dobj = info->dobj;
          this->resPri = info->resPri;
          this->resDual = info->resDual;
          this->relGap = info->relGap;
          this->solveTime = info->solveTime;
          this->setupTime = info->setupTime;
          this->status = (char *) malloc(strlen(info->status)+1);
          strcpy(this->status, info->status);
     }

     void freeSol(Sol *sol) {
          if (sol) {
               if (sol->x) {
                    free(sol->x);
                    sol->x = NULL;
               }
               if (sol->y) {
                    free(sol->y);
                    sol->y = NULL;
               }
               if (sol->s) {
                    free(sol->s);
                    sol->s = NULL;
               }
          }
     }
};
#endif