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

    dydt[1] = -(beta * I * S / N) + (gamma * I); // dS/dt
    dydt[2] = (beta * I * S / N) - (gamma * I); // dI/dt
    
    return dydt;
  }
  
  real incidR(real S, real I, real beta) {
    return beta * I * S / (S + I);
  }
  
  real recovR(real I, real gamma){
    return gamma * I;
  }
  
}

// Input data
data {
  int<lower=0> ntime;
  real<lower=0> new_cases[ntime]; // Inf at each time point
  real<lower=0> pop_sus[ntime+1]; // Obs possible susp
  real<lower=0> ts[ntime];
}

transformed data {
  int<lower=0> ntime_w0;
  ntime_w0 = ntime + 1;
}


parameters {
  vector<lower=0>[2] state0; // Initial Sus and Inf
  real<lower=0> s_sigma; // Overall var S
  real<lower=0> i_sigma; // Overall var I
  real<lower=0> beta; // Inf rate
  real<lower=0> gamma; // Recovery rate
}

transformed parameters {
  array[ntime_w0] vector<lower=0>[2] y; // Sus and Inf states after 0 time
  array[ntime] vector<lower=0>[2] rates; // Sus and Inf states after 0 time
  y[1,1] = state0[1];
  y[1,2] = state0[2];
  
  y[2:ntime_w0, 1:2] = ode_rk45(sis, state0, 0, ts, beta, gamma);
  
  // One step less
  for(t in 1:ntime) {
    rates[t, 1] = incidR(y[t, 1], y[t, 2], beta);
    rates[t, 2] = recovR(y[t, 2], gamma);
  }
}

model {
  // Priors
  state0[1] ~ lognormal(log(100), 3);
  state0[2] ~ lognormal(log(10), 1);
  s_sigma ~ exponential(1);
  i_sigma ~ exponential(1);
  beta ~ normal(1, 3);
  gamma ~ normal(4, 1.5); // Weigh on more than a few days recovery
  
  // Likelihood
  pop_sus ~ lognormal(log(y[,1]), s_sigma);
  new_cases ~ lognormal(log(rates[,1]), i_sigma);
}

generated quantities {
  real R0 = beta / gamma; // Rough estimate with assumptions on suscep of pop
  real recov_time = 1 / gamma;
  array[ntime_w0] real<lower=0> y_pred_s;
  array[ntime] real<lower=0> y_pred_i;
  y_pred_s = lognormal_rng(log(y[,1]), s_sigma);
  y_pred_i = lognormal_rng(log(rates[,1]), i_sigma);
}
