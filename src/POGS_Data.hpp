#ifndef POGS_DATA_H
#define POGS_DATA_H

#include <vector>
#include <string>
#include "pogs.h"
#include "pogs_fork/src/include/timer.h"

#ifdef FAO_GPU
#include "FAO_DAG.cuh"
#else
#include "FAO_DAG.hpp"
#endif

template <class T>
class POGS_Data {
public:
     T *c;
     size_t c_len;
     T *b;
     size_t b_len;
     T *Adata;
     int *Aindices;
     int *Aindptr;
     size_t nnz;
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

     T *x;
     T *y;

     ~POGS_Data() {
          free(status);
     }


     void load_c(T *c, int c_len) {
          this->c = c;
          this->c_len = c_len;
     }

     void load_b(T *b, int b_len) {
          this->b = b;
          this->b_len = b_len;
     }

     void load_x(T *x, int x_len) {
          this->x = x;
     }

     void load_y(T *y, int y_len) {
          this->y = y;
     }

     // void load_Adata(double *Adata, int nnz) {
     //      this->Adata = Adata;
     //      this->nnz = nnz;
     // }

     // void load_Aindices(int *Aindices, int nnz) {
     //      this->Aindices = Aindices;
     //      this->nnz = nnz;
     // }

     // void load_Aindptr(int *Aindptr, int nnz) {
     //      this->Aindptr = Aindptr;
     //      this->nnz = nnz;
     // }

     // EQ, LEQ, SOC, SOC_EW, SDP, EXP, BOOL, INT = range(8)
     Cone get_pogs_cone(int cone) {
          if (cone == 0) {
               return kConeZero;
          } else if (cone == 1) {
               return kConeNonNeg;
          } else if (cone == 2) {
               return kConeSoc;
          } else if (cone == 3) {
               return kConeSdp;
          } else if (cone == 4) {
               return kConeExpPrimal;
          }
     }

     // int mat_free_scs_solve(FAO_DAG* fao_dag,
     //           int f, int l, std::vector<int> q, std::vector<int> s,
     //           int ep, size_t max_iters, size_t equil_steps,
     //           size_t samples, bool precond, double eps,
     //           bool rand_seed) {
     //      scs::Cone * k;
     //      scs::Data * d;
     //      scs::Info info = { 0 };
     //      // Work * w;
     //      scs::Sol * sol;
     //      k = (scs::Cone *) calloc(1, sizeof(scs::Cone));
     //      d = (scs::Data *) calloc(1, sizeof(scs::Data));
     //      sol = (scs::Sol *) calloc(1, sizeof(scs::Sol));

     //      // Load cones.
     //      k->f = f;
     //      k->l = l;
     //      k->q = q.data();
     //      k->qsize = q.size();
     //      k->s = s.data();
     //      k->ssize = s.size();
     //      k->ep = ep;
     //      k->ed = 0;

     //      // Load c, b, cone info into data structures.
     //      d->b = b;
     //      d->c = c;
     //      d->m = b_len;
     //      d->n = c_len;
     //      d->fao_dag = (void *) fao_dag;
     //      d->Amul = fao_dag->static_forward_eval;
     //      d->ATmul = fao_dag->static_adjoint_eval;
     //      d->dag_input = fao_dag->get_forward_input()->data;
     //      d->dag_output = fao_dag->get_adjoint_input()->data;

     //      d->CG_RATE = 2.0;
     //      d->VERBOSE = true;
     //      d->MAX_ITERS = max_iters;
     //      d->EPS = eps;
     //      d->ALPHA = 1.5;
     //      d->RHO_X = 1e-3;
     //      d->SCALE = 5;
     //      d->NORMALIZE = true; /* boolean, heuristic data rescaling: 1 */
     //      d->WARM_START = false; /* boolean, warm start (put initial guess in Sol struct): 0 */
     //      d->EQUIL_STEPS = equil_steps; /* how many steps of equilibration to take: 1 */
     //      d->EQUIL_P = 2;  /* Which Lp norm equilibration used? */
     //      d->EQUIL_GAMMA = 0; /* Regularization parameter for equilibration: 1e-8 */
     //      d->STOCH = true; /* Use random approximation of L2 norms? */
     //      d->SAMPLES = samples; /* Number of samples for approximation: 200 */
     //      d->PRECOND = precond; /* boolean, use preconditioner for CG? */
     //      d->RAND_SEED = rand_seed;

     //      scs::scs(d, k, sol, &info);
     //      load_info(&info);
     //      memcpy(this->x, sol->x, sizeof(double)*this->c_len);
     //      memcpy(this->y, sol->y, sizeof(double)*this->b_len);
     //      free(d);
     //      free(k);
     //      freeSol(sol);
     //      return 0;
     // }

     /* function: pogs_solve()
     *
     */
     double mat_free_solve(FAO_DAG<T>* fao_dag,
               std::vector< std::pair<int, std::vector<int> > >& cones,
                  double rho,
                  bool verbose,
                  double abs_tol,
                  double rel_tol,
                  int max_iter,
                  size_t samples,
                  size_t equil_steps) {

          // printf("size of cones=%d\n", cones.size());
          std::vector<ConeConstraint> Kx, Ky;
          set_cones(cones, Kx, Ky);

          auto Amul = fao_dag->static_forward_eval;
          auto ATmul = fao_dag->static_adjoint_eval;
          T *dag_input = fao_dag->get_forward_input()->data;
          T *dag_output = fao_dag->get_adjoint_input()->data;

          pogs::MatrixFAO<T> A_(dag_output, b_len,
                                dag_input, c_len,
                                Amul, ATmul,
                                (void *) fao_dag,
                                samples,
                                equil_steps);
          pogs::PogsIndirectCone<T, pogs::MatrixFAO<T> > pogs_data(A_, Kx, Ky);
          // double t = timer<double>();
          if (verbose) {
               pogs_data.SetVerbose(5);
          } else {
               pogs_data.SetVerbose(0);
          }
          pogs_data.SetRho(rho);
          pogs_data.SetAbsTol(abs_tol);
          pogs_data.SetRelTol(rel_tol);
          pogs_data.SetMaxIter(max_iter);
          std::vector<T> c_vec(c, c + c_len);
          std::vector<T> b_vec(b, b + b_len);
          double t = timer<double>();
          pogs_data.Solve(b_vec, c_vec);
          this->solveTime = timer<double>() - t;
          // Copy out results.
          for (size_t i=0; i < c_len; i++) {
               x[i] = pogs_data.GetX()[i];
          }
          for (size_t i=0; i < b_len; i++) {
              y[i] = pogs_data.GetMu()[i];
          }
          return pogs_data.GetOptval();
     }


     void set_cones(std::vector< std::pair<int, std::vector<int> > >& cones,
                    std::vector<ConeConstraint>& Kx,
                    std::vector<ConeConstraint>& Ky) {
          for (size_t i=0; i < cones.size(); i++) {
               Cone cone_type = get_pogs_cone(cones[i].first);
               // printf("cone type=%d\n", cone_type);
               std::vector<int> orig_idx = cones[i].second;
               std::vector<CONE_IDX> idx;
               for (size_t j=0; j < orig_idx.size(); j++) {
                    // printf("cone idx %d=%d\n", j, orig_idx[j]);
                    idx.push_back((CONE_IDX) orig_idx[j]);
               }
               Ky.emplace_back(cone_type, idx);
          }
     }

     /* function: pogs_solve()
     *
     */
     double solve(int nnz,
               std::vector<T>& Adata,
               std::vector<int>& Aindices,
               std::vector<int>& Aindptr,
               std::vector< std::pair<int, std::vector<int> > >& cones,
                  double rho,
                  bool verbose,
                  double abs_tol,
                  double rel_tol,
                  int max_iter) {

          // printf("size of cones=%d\n", cones.size());
          std::vector<ConeConstraint> Kx, Ky;
          set_cones(cones, Kx, Ky);

          pogs::MatrixSparse<T> A_('c', b_len, c_len, nnz, Adata.data(), Aindptr.data(), Aindices.data());
          pogs::PogsIndirectCone<T, pogs::MatrixSparse<T> > pogs_data(A_, Kx, Ky);
          // pogs::MatrixDense<double> A_('r', m, n, Adense.data());
          // pogs::PogsDirectCone<double, pogs::MatrixDense<double> > pogs_data(A_, Kx, Ky);
          // double t = timer<double>();
          if (verbose) {
               pogs_data.SetVerbose(5);
          } else {
               pogs_data.SetVerbose(0);
          }
          pogs_data.SetRho(rho);
          pogs_data.SetAbsTol(abs_tol);
          pogs_data.SetRelTol(rel_tol);
          pogs_data.SetMaxIter(max_iter);
          std::vector<T> c_vec(c, c + c_len);
          std::vector<T> b_vec(b, b + b_len);
          double t = timer<double>();
          pogs_data.Solve(b_vec, c_vec);
          this->solveTime = timer<double>() - t;
          // Copy out results.
          for (size_t i=0; i < c_len; i++) {
               x[i] = pogs_data.GetX()[i];
          }
          for (size_t i=0; i < b_len; i++) {
              y[i] = pogs_data.GetMu()[i];
          }
          return pogs_data.GetOptval();
     }

     // void load_info(scs::Info *info) {
     //      this->statusVal = info->statusVal;
     //      this->iter = info->iter;
     //      this->cgIter = info->cgIter;
     //      this->pobj = info->pobj;
     //      this->dobj = info->dobj;
     //      this->resPri = info->resPri;
     //      this->resDual = info->resDual;
     //      this->relGap = info->relGap;
     //      this->solveTime = info->solveTime;
     //      this->setupTime = info->setupTime;
     //      this->status = (char *) malloc(strlen(info->status)+1);
     //      strcpy(this->status, info->status);
     // }

     // void freeSol(scs::Sol *sol) {
     //      if (sol) {
     //           if (sol->x) {
     //                free(sol->x);
     //                sol->x = NULL;
     //           }
     //           if (sol->y) {
     //                free(sol->y);
     //                sol->y = NULL;
     //           }
     //           if (sol->s) {
     //                free(sol->s);
     //                sol->s = NULL;
     //           }
     //      }
     // }
};

#ifdef SWIG
%template(POGS_Datad) POGS_Data<double>;
%template(POGS_Dataf) POGS_Data<float>;
#endif

#endif
