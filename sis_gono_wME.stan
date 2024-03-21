/*

SIS Model Gonorrhea W/ ME

Including additional params on Measurement Error from surveillance data

*/




// Function for SIS (states in and returned must be same length)
functions {

  // Using more recent syntax for ODE solvers in STAN
  vector sis(real t, vector state, real beta, real gamma) {
    
    vector[2] dydt;
    
    real S = state[1];
    real I = state[2];
    real N = S + I;
    real incid = beta * I * S / N;
    real recov = gamma * I;

    dydt[1] = -(incid) + recov; // dS/dt
    dydt[2] = incid - recov; // dI/dt

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
  real<lower=0> new_cases[ntime]; // New Infs
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
  real<lower=0> p_s; 
  real<lower=0,upper=1> p_i; // Surveillance prop
}

transformed parameters {
  array[ntime_w0] vector<lower=0>[2] y; // Sus and Inf states after 0 time
  array[ntime] vector<lower=0>[2] rates; // Sus and Inf states after 0 time
  y[1,1] = state0[1]; // S0
  y[1,2] = state0[2]; // I0
  
  // time 1 is 0, for incid, need to extend so can all be in 1 array
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
  beta ~ normal(1, 3); // Recall its dt per quarter
  gamma ~ normal(4, 1.5); // Weigh on more than a few days recovery (dt is quarter... ~1/0.25)
  p_s ~ normal(2, 3); // Truncated normal 
  p_i ~ beta(40, 200);
  
  // Likelihood (loop otherwise STAN doesnt know how to multiply)
  for (t in 1:ntime_w0) { 
    pop_sus[t] ~ lognormal(log(y[t,1] * p_s), s_sigma);
    
    // change in incid is 1 less in size than all suscep
    if (t < ntime_w0) {
      new_cases[t] ~ lognormal(log(rates[t,1] * p_i), i_sigma);
    }
    
  }
}

generated quantities {
  real R0 = beta / gamma; // Rough estimate with assumptions on suscep of pop
  //vector[ntime] Reff; // S * R0, where S is prop suscpep at given time
  real recov_time = 1 / gamma;
  
  array[ntime_w0] real<lower=0> y_pred_s;
  array[ntime] real<lower=0> y_pred_i;
  
  vector[ntime_w0] log_likS;
  vector[ntime] log_likI;
  vector[ntime+ntime_w0] log_lik;
  
  for (t in 1:ntime_w0) { 
    y_pred_s[t] = lognormal_rng(log(y[t,1]* p_s), s_sigma);
    log_likS[t] = lognormal_lpdf(pop_sus[t] | log(y[t,1]*p_s), s_sigma);
    if (t < ntime_w0) {
      y_pred_i[t] = lognormal_rng(log(rates[t,1]* p_i), i_sigma);
      log_likI[t] = lognormal_lpdf(new_cases[t] | log(rates[t,1]*p_i), i_sigma);
    }
  }
  
  log_lik = append_row(log_likS, log_likI); // Combined loglik
  
}
