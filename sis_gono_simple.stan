//
// SIS Model Gonorrhea
//


// Function for SIS
functions {

  // Using more recent syntax for ODE solvers in STAN
  vector sis(real t, vector state, real beta, real gamma) {
    
    vector[2] dydt;
    
    real S = state[1];
    real I = state[2];
    real N = S + I;

    dydt[1] = -beta * I * S / N + gamma * I; // dS/dt
    dydt[2] = beta * I * S / N - gamma * I; // dI/dt
    
    return dydt;
  }
  
}

// Input data
data {
  int<lower=0> ntime;
  real<lower=0> cases[ntime]; // Inf at each time point
  real<lower=0> ts[ntime-1];
}


parameters {
  vector<lower=0, upper = 5000000>[2] state0; // Initial Sus and Inf
  real<lower=0> sigma; // Overall var
  real<lower=0> beta; // Inf rate
  real<lower=0> gamma; // Recovery rate
  //real<lower=0,upper=1> p; // Surveillance prop
}

transformed parameters {
  array[ntime] vector<lower=0>[2] y; // Sus and Inf states after 0 time
  y[1,1] = state0[1];
  y[1,2] = state0[2];
  y[2:ntime, 1:2] = ode_rk45(sis, state0, 0, ts, beta, gamma);
}

model {
  // Priors
  state0[1] ~ lognormal(log(1e6), 3);
  state0[2] ~ lognormal(log(10), 1);
  sigma ~ exponential(1);
  beta ~ normal(0.25, 2);
  gamma ~ normal(0.3, 0.5); // Weigh on more than a few days recovery
  //p ~ beta();
  
  // Likelihood
  cases ~ lognormal(log(y[,2]), sigma);
}

