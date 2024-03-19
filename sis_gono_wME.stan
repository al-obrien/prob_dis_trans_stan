//
// SIS Model Gonorrhea
//

// Including additional params on Measurement Error from surveillance data


// Function for SIS
functions {

  // Using more recent syntax for ODE solvers in STAN
  vector sis(real t, vector state, real beta, real gamma) {
    
    vector[2] dydt;
    
    real S = state[1];
    real I = state[2];
    real N = S + I;

    dydt[1] = -(beta * I * S / N) + (gamma * I); // dS/dt
    dydt[2] = (beta * I * S / N) - (gamma * I); // dI/dt
    
    return dydt;
  }
  
}

// Input data
data {
  int<lower=0> ntime;
  real<lower=0> cases[ntime]; // Inf at each time point
  real<lower=0> pop_sus[ntime]; // Obs possible susp
  real<lower=0> ts[ntime-1];
}


parameters {
  vector<lower=0>[2] state0; // Initial Sus and Inf
  real<lower=0> s_sigma; // Overall var S
  real<lower=0> i_sigma; // Overall var I
  real<lower=0> beta; // Inf rate
  real<lower=0> gamma; // Recovery rate
  real<lower=0> p_s; 
  real<lower=0,upper=1> p_i; // Surveillance prop
}

transformed parameters {
  array[ntime] vector<lower=0>[2] y; // Sus and Inf states after 0 time
  y[1,1] = state0[1];
  y[1,2] = state0[2];
  y[2:ntime, 1:2] = ode_rk45(sis, state0, 0, ts, beta, gamma);
}

model {
  // Priors
  state0[1] ~ lognormal(log(100), 3);
  state0[2] ~ lognormal(log(10), 1);
  s_sigma ~ exponential(1);
  i_sigma ~ exponential(1);
  beta ~ normal(0.1, 4);
  gamma ~ normal(0.3, 0.5); // Weigh on more than a few days recovery
  p_s ~ normal(1, 5); // Truncated normal 
  p_i ~ beta(40, 200);
  
  // Likelihood
  for ( t in 1:ntime) { // Loop otherwise STAN doesnt know how to multiply
    pop_sus ~ lognormal(log(y[t,1] * p_s), s_sigma);
    cases ~ lognormal(log(y[t,2] * p_i), i_sigma);
  }

}

generated quantities {
  array[ntime] real<lower=0> y_pred_i;
  array[ntime] real<lower=0> y_pred_s;
  for (t in 1:ntime) { 
    y_pred_s = lognormal_rng(log(y[t,1]* p_s), s_sigma);
    y_pred_i = lognormal_rng(log(y[t,2]* p_i), i_sigma);
  }
}
