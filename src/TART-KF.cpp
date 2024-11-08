#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;

//' Estimating state vector by TART Kalman filter
//' 
//' @param a1 mean of initial state vector
//' @param P1 covariance matrix of initial state vector
//' @param T transition matrix of state vector
//' @param Z transition matrix of state vector
//' @param Q covariance matrix of state noise vector
//' @param H covariance matrix of observation noise vector
//' @param y observation vectors
//' @param lambda_init initial regularization parameter of adaptive ridge
//' @param S number of times to perform ridge estimation in adaptive ridge
//' @param beta1 parameter of Adam
//' @param beta2 parameter of Adam
//' @param eta learning rate of Adam
//' @param c_e interval over which the average is taken for M_t in TART-KF
//' @param c_F interval over which the average is taken for F_t in TART-KF
//' @param d_F rate of lambda control by Kalman filter
//' @param delta_AR small value to prevent dividing by 0 in adaptive ridge
//' @param delta_g value to approximate derivative of L_t for Adam
//' @param delta_v small value to prevent dividing by 0 in Adam
//'
//' @export
// [[Rcpp::export]]
List TART_KF(arma :: vec a1, arma :: mat P1, arma :: mat T, arma :: cube Z, arma :: mat Q, arma :: mat H, arma :: mat y, double lambda_init = 0.0, int S = 3, double beta1 = 0.9, double beta2 = 0.999, double eta = 0.002, int c_e = 100, int c_F = 5, double d_F = 1.0, double delta_AR = 1e-8, double delta_g = 0.01, double delta_v = 1e-8) {
  int N = y.n_cols;
  int d = y.n_rows;
  int p = a1.n_elem;
  
  arma :: mat at(p,N+1);
  arma :: mat att(p,N);
  
  arma :: mat at_ART(p,N+1);
  arma :: mat att_ART(p,N);
  
  arma :: cube Pt(p,p,N+1);
  arma :: cube Ptt(p,p,N+1);
  
  at.col(0) = a1;
  at_ART.col(0) = a1;
  Pt.slice(0) = P1;
  
  arma :: mat vt_ART(d,N);
  arma :: mat SFE0_ART(d,N);
  
  double lambda = lambda_init;
  arma :: vec lambda_all(N);
  
  lambda_all(0) = lambda;

  arma :: vec acc_ind_log(c_F);
  
  double moment1 = 0;
  double moment2 = 0;
  
  double mean_update = lambda_init/10;
  arma :: vec all_update;
  
  vt_ART.col(0) = y.col(0) - Z.slice(0) * at_ART.col(0);
  
  for (int t = 0; t < N; t++){
    arma :: mat Pt_inv = inv(Pt.slice(t));
    arma :: mat A = trans(Z.slice(t)) * inv(H) * Z.slice(t) + Pt_inv;
    arma :: mat K = Pt.slice(t) * trans(Z.slice(t)) * inv( Z.slice(t) * Pt.slice(t) * trans(Z.slice(t)) + H );
    
    att.col(t) = at.col(t) + K * (y.col(t) - Z.slice(t) * at.col(t));
    at.col(t+1) = T * att.col(t);
    
    Ptt.slice(t) = Pt.slice(t) - K * ( Z.slice(t) * Pt.slice(t) * trans(Z.slice(t)) + H ) * trans(K);
    Pt.slice(t+1) = T * Ptt.slice(t) * trans(T) + Q;
    
    arma :: vec tmp = trans(Z.slice(t)) * inv(H) * y.col(t) + Pt_inv * at_ART.col(t);
    att_ART.col(t) = inv( A + lambda * eye(p, p)) * tmp;
    for (int i = 0; i < S; i++){
      arma :: mat D = diagmat( lambda/(square(att_ART.col(t)) + delta_AR) );
      att_ART.col(t) = inv( A + D) * tmp;
    }
    at_ART.col(t+1) = T * att_ART.col(t);
    
    if(t == N-1){
      break;
    }
    
    vt_ART.col(t+1) = y.col(t+1) - Z.slice(t+1) * at_ART.col(t+1);
    arma :: vec vt_KF = y.col(t+1) - Z.slice(t+1) * at.col(t+1);
    SFE0_ART.col(t+1) = square( vt_ART.col(t+1) );
    
    arma :: mat M = diagmat(1 / (mean(SFE0_ART.submat(0,std::max(0,t-c_e),d-1,t+1),1) + delta_AR )  );
    
    double SFE_ART = as_scalar(trans(vt_ART.col(t+1)) * M * vt_ART.col(t+1))/d;
    double SFE_KF = as_scalar(trans(vt_KF) * M * vt_KF)/d;
    
    if( SFE_ART > SFE_KF ){
      lambda -= ( SFE_ART - SFE_KF )*mean_update*d_F*( 1+sum(acc_ind_log) );
      lambda = std::max(lambda,0.0);

      acc_ind_log.shed_row(0);
      acc_ind_log.resize(c_F);
      acc_ind_log(c_F - 1) = 1;
      
      lambda_all[t+1] = lambda;
      continue;
    }
    
    double lambda_up = lambda + delta_g;
    arma :: vec att_ART_up = inv( A + lambda_up * eye(p, p)) * tmp;
    for (int i = 0; i < S; i++){
      arma :: mat D = diagmat( lambda_up/(square(att_ART_up) + delta_AR) );
      att_ART_up = inv( A + D) * tmp;
    }
    arma :: vec at_ART_up = T * att_ART_up;
    arma :: vec e_up = y.col(t+1) - Z.slice(t+1) * at_ART_up;
    double SFE_up = as_scalar(trans(e_up) * M * e_up)/d;
    
    double lambda_down = std::max(0.0,lambda - delta_g);
    arma :: vec att_ART_down = inv( A + lambda_down * eye(p, p)) * tmp;
    for (int i = 0; i < S; i++){
      arma :: mat D = diagmat( lambda_down/(square(att_ART_down) + delta_AR) );
      att_ART_down = inv( A + D) * tmp;
    }
    arma :: vec at_ART_down = T * att_ART_down;
    arma :: vec e_down = y.col(t+1) - Z.slice(t+1) * at_ART_down;
    double SFE_down = as_scalar(trans(e_down) * M * e_down)/d;
      
    double G_lambda = (SFE_up - SFE_down) / (lambda_up-lambda_down);
    moment1 = beta1*moment1 + (1 - beta1)*G_lambda;
    moment2 = beta2*moment2 + (1 - beta2)*(pow(G_lambda,2));
        
    double moment1_hat = moment1/(1-std::pow(beta1,t+1) );
    double moment2_hat = moment2/(1-std::pow(beta2,t+1) );
        
    double update = eta*moment1_hat/(std::sqrt(moment2_hat) + delta_v );
    lambda = lambda - update;
    lambda = std::max(0.0,lambda);

    all_update.resize(all_update.n_elem + 1);
    all_update(all_update.n_elem - 1) = std::abs(update);
    mean_update = mean(all_update);
    
    acc_ind_log.shed_row(0);
    acc_ind_log.resize(c_F);
    acc_ind_log(c_F - 1) = 0;
    
    lambda_all[t+1] = lambda;
  }
  
  List res;
  res["at"] = at_ART;
  res["att"] = att_ART;
  res["vt"] = vt_ART;
  res["lambda"] = lambda_all;
  
  return res;
}
