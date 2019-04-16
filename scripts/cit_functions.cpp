#include <armadillo>
#include <iostream>
#include <Python.h>
typedef arma::mat Mat;

auto linreg_with_stats(Mat y, Mat X){
  double n = y.n_rows;
  Mat beta = arma::pinv(X.t() * X) * X.t() * y;
  double y_bar = arma::accu(y) / n;
  double TSS = arma::accu(arma::square(y - y_bar));
  double RSS = arma::accu(arma::square(y - (X * beta)));
  Mat var_beta = RSS * arma::pinv(X.t() * X);
  Mat se = arma::sqrt(var_beta.diag());
  Mat t = beta / se;
  struct result {Mat beta; double RSS; double TSS; Mat se; Mat t;};
  return result {beta,RSS,TSS,se,t};
}

extern "C"{
  void linreg()
}


int main(int argc, char const *argv[]) {
  Mat A = arma::randu<arma::mat>(5,5);
  Mat b = arma::sum(A,1);
  auto result = linreg_with_stats(b,A);
  std::cout << result.beta << std::endl;
  return 0;
}
