#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;

//' Estimating state vector by TART Kalman filter
//' 
//' @param a1 p * 1 mean of initial state vector
//' @param P1 p * p covariance matrix of initial state vector
//' @param T p * p * n (or p * p * n_T, if T has an n_T cycle) array giving transition matrix in state equation for each time
//' @param Z d * p * n (or p * p * n_Z, if Z has an n_Z cycle) array giving transition matrix in observation equation for each time
//' @param Sigma_eta p * p * n (or p * p * n_Sigma_eta, if Sigma_eta has an n_Sigma_eta cycle) array giving covariance matrix of state noise vector for each time
//' @param Sigma_epsilon d * d * n (or d * d * n_Sigma_epsilon, if Sigma_epsilon has an n_Sigma_epsilon cycle) array giving covariance matrix of observation noise vector for each time
//' @param y d * n observation vector for each time
//' @param lambda_init initial regularization parameter of adaptive ridge
//' @param S number of times to perform ridge estimation in adaptive ridge
//' @param beta1 parameter of Adam
//' @param beta2 parameter of Adam
//' @param gamma learning rate of Adam
//' @param c_e interval over which the average is taken for M_t in TART-KF
//' @param c_F interval over which the average is taken for F_t in TART-KF
//' @param d_F rate of lambda control by Kalman filter
//' @param delta_AR small value to prevent dividing by 0 in adaptive ridge
//' @param delta_g value to approximate derivative of L_t for Adam
//' @param delta_v small value to prevent dividing by 0 in Adam
//'
//' @export
// [[Rcpp::export]]
List TART_KF(arma :: vec a1, arma :: mat P1, arma :: cube T, arma :: cube Z, arma :: cube Sigma_eta, arma :: cube Sigma_epsilon, arma :: mat y, double lambda_init = 0.0, int S = 3, double beta1 = 0.9, double beta2 = 0.999, double gamma = 0.002, int c_e = 100, int c_F = 5, double d_F = 1.0, double delta_AR = 1e-8, double delta_g = 0.01, double delta_v = 1e-8) {
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

  int n_T = T.n_slices;
  int n_Z = Z.n_slices;
  int n_Sigma_eta = Sigma_eta.n_slices;
  int n_Sigma_epsilon = Sigma_epsilon.n_slices;

  for (int t = 0; t < N; t++){
    arma :: mat Tt = T.slice(t % n_T);
    arma :: mat Zt = Z.slice(t % n_Z);
    arma :: mat Zt_1 = Z.slice( (t+1) % n_Z);
    arma :: mat Sigma_etat = Sigma_eta.slice(t % n_Sigma_eta);
    arma :: mat Sigma_epsilont = Sigma_epsilon.slice(t % n_Sigma_epsilon);
    
    arma :: mat Pt_inv = inv(Pt.slice(t));
    arma :: mat Sigma_epsilont_inv = inv(Sigma_epsilont);
    arma :: mat A = trans(Zt) * Sigma_epsilont_inv * Zt + Pt_inv;
    arma :: mat K = Pt.slice(t) * trans(Zt) * inv( Zt * Pt.slice(t) * trans(Zt) + Sigma_epsilont );
    
    att.col(t) = at.col(t) + K * (y.col(t) - Zt * at.col(t));
    at.col(t+1) = Tt * att.col(t);
    
    Ptt.slice(t) = Pt.slice(t) - K * ( Zt * Pt.slice(t) * trans(Zt) + Sigma_epsilont ) * trans(K);
    Pt.slice(t+1) = Tt * Ptt.slice(t) * trans(Tt) + Sigma_etat;
    
    arma :: vec tmp = trans(Zt) * Sigma_epsilont_inv * y.col(t) + Pt_inv * at_ART.col(t);
    att_ART.col(t) = inv( A + lambda * eye(p, p)) * tmp;
    for (int i = 0; i < S; i++){
      arma :: mat D = diagmat( lambda/(square(att_ART.col(t)) + delta_AR) );
      att_ART.col(t) = inv( A + D) * tmp;
    }
    at_ART.col(t+1) = Tt * att_ART.col(t);
    
    if(t == N-1){
      break;
    }
    
    vt_ART.col(t+1) = y.col(t+1) - Zt_1 * at_ART.col(t+1);
    arma :: vec vt_KF = y.col(t+1) - Zt_1 * at.col(t+1);
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
    arma :: vec at_ART_up = Tt * att_ART_up;
    arma :: vec e_up = y.col(t+1) - Zt_1 * at_ART_up;
    double SFE_up = as_scalar(trans(e_up) * M * e_up)/d;
    
    double lambda_down = std::max(0.0,lambda - delta_g);
    arma :: vec att_ART_down = inv( A + lambda_down * eye(p, p)) * tmp;
    for (int i = 0; i < S; i++){
      arma :: mat D = diagmat( lambda_down/(square(att_ART_down) + delta_AR) );
      att_ART_down = inv( A + D) * tmp;
    }
    arma :: vec at_ART_down = Tt * att_ART_down;
    arma :: vec e_down = y.col(t+1) - Zt_1 * at_ART_down;
    double SFE_down = as_scalar(trans(e_down) * M * e_down)/d;
      
    double G_lambda = (SFE_up - SFE_down) / (lambda_up-lambda_down);
    moment1 = beta1*moment1 + (1 - beta1)*G_lambda;
    moment2 = beta2*moment2 + (1 - beta2)*(pow(G_lambda,2));
        
    double moment1_hat = moment1/(1-std::pow(beta1,t+1) );
    double moment2_hat = moment2/(1-std::pow(beta2,t+1) );
        
    double update = gamma*moment1_hat/(std::sqrt(moment2_hat) + delta_v );
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
