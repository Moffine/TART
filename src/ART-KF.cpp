#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;

//' Estimating state vector by ART Kalman filter
//' 
//' @param a1 mean of initial state vector
//' @param P1 covariance matrix of initial state vector
//' @param T transition matrix of state vector
//' @param Z transition matrix of state vector
//' @param Q covariance matrix of state noise vector
//' @param H covariance matrix of observation noise vector
//' @param y observation vectors
//' @param lambda regularization parameter of adaptive ridge
//' @param S number of times to perform ridge estimation in adaptive ridge
//' @param delta_AR small value to prevent dividing by 0 in adaptive ridge
//'
//' @export
// [[Rcpp::export]]
List ART_KF(arma :: vec a1, arma :: mat P1, arma :: cube T, arma :: cube Z, arma :: cube Q, arma :: cube H, arma :: mat y, double lambda, int S = 3, double delta_AR = 1e-8) {
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
  int n_Q = Q.n_slices;
  int n_H = H.n_slices;

  for (int t = 0; t < N; t++){
    arma :: mat Tt = T.slice(t % n_T);
    arma :: mat Zt = Z.slice(t % n_Z);
    arma :: mat Zt_1 = Z.slice( (t+1) % n_Z);
    arma :: mat Qt = Q.slice(t % n_Q);
    arma :: mat Ht = H.slice(t % n_H);
    
    arma :: mat Pt_inv = inv(Pt.slice(t));
    arma :: mat Ht_inv = inv(Ht);
    arma :: mat A = trans(Zt) * Ht_inv * Zt + Pt_inv;
    arma :: mat K = Pt.slice(t) * trans(Zt) * inv( Zt * Pt.slice(t) * trans(Zt) + Ht );
    
    att.col(t) = at.col(t) + K * (y.col(t) - Zt * at.col(t));
    at.col(t+1) = Tt * att.col(t);
    
    Ptt.slice(t) = Pt.slice(t) - K * ( Zt * Pt.slice(t) * trans(Zt) + Ht ) * trans(K);
    Pt.slice(t+1) = Tt * Ptt.slice(t) * trans(Tt) + Qt;
    
    arma :: vec tmp = trans(Zt) * Ht_inv * y.col(t) + Pt_inv * at_ART.col(t);
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
