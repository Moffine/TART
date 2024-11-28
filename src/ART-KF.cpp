#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;

//' Estimating state vector by ART Kalman filter
//' 
//' @param a1 p * 1 mean of initial state vector
//' @param P1 p * p covariance matrix of initial state vector
//' @param T p * p * n (or p * p * n_T, if T has an n_T cycle) array giving transition matrix in state equation for each time
//' @param Z d * p * n (or p * p * n_Z, if Z has an n_Z cycle) array giving transition matrix in observation equation for each time
//' @param Sigma_eta p * p * n (or p * p * n_Sigma_eta, if Sigma_eta has an n_Sigma_eta cycle) array giving covariance matrix of state noise vector for each time
//' @param Sigma_epsilon d * d * n (or d * d * n_Sigma_epsilon, if Sigma_epsilon has an n_Sigma_epsilon cycle) array giving covariance matrix of observation noise vector for each time
//' @param y d * n observation vector for each time
//' @param lambda regularization parameter of adaptive ridge
//' @param S number of times to perform ridge estimation in adaptive ridge
//' @param delta_AR small value to prevent dividing by 0 in adaptive ridge
//'
//' @export
// [[Rcpp::export]]
List ART_KF(arma :: vec a1, arma :: mat P1, arma :: cube T, arma :: cube Z, arma :: cube Sigma_eta, arma :: cube Sigma_epsilon, arma :: mat y, double lambda, int S = 3, double delta_AR = 1e-8) {
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
  arma :: mat SFE_ART(d,N);

  arma :: vec lambda_all(N);
  
  lambda_all(0) = lambda;
  
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
    SFE_ART.col(t+1) = square( vt_ART.col(t+1) );
    arma :: vec SFE_KF = square( y.col(t+1) - Zt_1 * at.col(t+1) );
    lambda_all[t+1] = lambda;
  }
  
  List res;
  res["at"] = at_ART;
  res["att"] = att_ART;
  res["vt"] = vt_ART;
  res["lambda"] = lambda_all;
  
  return res;
}
