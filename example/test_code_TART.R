#############download package#######################
devtools::install_github("Moffine/TART")
library(TART)

#############generate simulation data#######################
set.seed(1)

N <- 5000
y <- numeric(N)
alpha <- numeric(N)

for(t in 1:N){
  alpha[t] <- max(0,4 - ((1000-t)/1000)^2 )
  y[t] <- alpha[t] + rnorm(1,0,1)
}

T <- matrix(1,1,1)
Z <- array(1,dim=c(1,1,N))
Q <- matrix(0.0001,1,1)

H <- matrix(1,1,1)

#############estimate state variable (alpha) by ART-KF#######################
res_ART <- TART::ART_KF(a1=rep(0,1), P1=matrix(1,1,1)*1000, T = T, Z = Z, Q = Q, H = H, y = matrix(y,1,N), lambda=0.1)

plot(y, type="l",lwd=1,cex.axis = 1.5,xlab="",ylab="") #plot of observations(black line)
lines(res_ART$att[,1:N], type="l",lwd=3,col="red") #plot of estimates of ART_KF(red line)
lines(alpha, lwd=2,col="green", type="l",lty=2) #plot of true state variables(green line)

#############estimate state variable (alpha) by TART-KF######################
res_TART <- TART::TART_KF(a1=rep(0,1), P1=matrix(1,1,1)*1000, T = T, Z = Z, Q = Q, H = H, y = matrix(y,1,N) )

plot(y, type="l",lwd=1,cex.axis = 1.5,xlab="",ylab="") #plot of observations(black line)
lines(res_TART$att[,1:N], type="l",lwd=3,col="red") #plot of estimates of TART_KF(red line)
lines(alpha, lwd=2,col="green", type="l",lty=2) #plot of true state variables(green line)

plot(res_TART$lambda[1:N],col="blue",type="l",pch=20,lwd=3,cex.axis = 1.5,xlab="",ylab="") #plot of lambda_t
