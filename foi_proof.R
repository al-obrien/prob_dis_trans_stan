beta = 0.8
N = 100
alpha = .2
Nv = c(alpha*N, (1-alpha) * N)
Iv = c(Nv[1] * .4, Nv[2] * .8)
Sv <- Nv - Iv
  
contact <- c(2, 6)
prob <- matrix(c(0.75, 0.25, 0.25, 0.75), nrow = 2)


# Contact matrix
W <-  prob * contact

# FOI
foi <- beta * W %*% (Iv/Nv)

# Amount trans
foi * Sv


# W/O Matrices
phi <- rep(NA, 2)
for(i in 1:2) {

  vec <- rep(NA, 2)
  for(j in 1:2) {
    vec[j] <- beta * prob[i,j] * contact[i] * Iv[j]/Nv[j]
  }
  phi[i] <- sum(vec)
}

# (beta * prob[1,1] * contact[1] * Iv[1]/Nv[1]) + (beta * prob[1,2] * contact[1] * Iv[2]/Nv[2])
# beta* 0.6 + beta*0.4

# Check
all(foi == phi)
