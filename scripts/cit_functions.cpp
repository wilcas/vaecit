#include <armadillo>
#include <iostream>
#include <tuple>

extern "C" {
  double linreg_get_t(int n, int p , double *y, double *X, int t_idx){
    arma::mat data = arma::mat(X,p,n,false,true).t();
    arma::mat response = arma::mat(y,n,1,false,true);
    arma::mat beta = arma::solve(data, response);
    double y_bar = arma::accu(response) / n;
    double TSS = arma::accu(arma::square(response - y_bar));
    double RSS = arma::accu(arma::square(response- (data * beta)));
    arma::mat var_beta = RSS * arma::pinv(data.t() * data);
    arma::mat se = arma::sqrt(var_beta.diag());
    arma::mat t = beta / se;
    return t(t_idx,0);
  }


  void linreg_with_stats(int n, int p , double *y, double *X, double  *beta,
    double &RSS, double &TSS, double *se, double *t){
    arma::mat data = arma::mat(X,p,n,false,true).t();
    arma::mat response = arma::mat(y,n,1,false,true);
    arma::mat tmp_beta = arma::solve(data,response);
    double y_bar = arma::accu(response) / n;

    TSS = arma::accu(arma::square(response - y_bar));
    RSS = arma::accu(arma::square(response- (data * tmp_beta)));

    arma::mat var_beta = RSS * arma::pinv(data.t() * data);
    arma::mat tmp_se = arma::sqrt(var_beta.diag());

    arma::mat tmp_tstat = tmp_beta / tmp_se;
    for(int i = 0; i < p; i++){
      se[i] = tmp_se(i,0);
      beta[i] = tmp_beta(i,0);
      t[i] = tmp_tstat(i,0);
    }
    return;
  }


  void run_bootstraps(int n, int num_bootstrap, double *T,
    double *residual, double *L, double *tstats){
    arma::mat tmp_residual = arma::join_rows(
      arma::ones<arma::mat>(n,1),
      arma::mat(residual,n,1,false,true)
    );
    arma::mat tmp_L = arma::mat(L,n,1,false,true);
    //initialize regression results
    double *resid;
    for(int i = 0; i < num_bootstrap; i++){
        //residual includes intercept
        tmp_residual = arma::shuffle(tmp_residual, 0);
        arma::mat design = arma::join_rows(
          tmp_residual,
          tmp_L
        );
        resid = design.memptr();
        tstats[i] = linreg_get_t(n,3,T,resid,1);
    }
    return;
  }
}


/* For testing C++ code */
int main(){
  double X  [] = {2,2,2};
  double resid [] = {5,6,7};
  double y [] = {4,4,4};
  double t [100];
  double RSS, TSS;

  run_bootstraps(3,100,y,resid,X,t);
  for(int i; i < 100; i++)
  std::cout << t[i] << std::endl;

  return 0;
}
