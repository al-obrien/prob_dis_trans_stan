/*

SIS Model Gonorrhea

Including additional params on Measurement Error from surveillance data and 
two types of contact rates (core grp vs other).


*/




// Function for SIS (states in and returned must be same length)
functions {
  
  vector sis(real t, vector state, matrix probM, vector c, real beta, real gamma) {
    
    vector[4] dydt;
    
    // Initial predicted states
    vector[2] S = state[1:2]; // Lo, Hi
    vector[2] I = state[3:4]; // Lo, Hi
    vector[2] N = S + I;
    
    // FOI, one for LO other for HI (phi[1] = FOI for lo, phi[2 = FOI for hi])
    vector[2] phi = (beta * c) .* to_vector(probM * to_matrix(I ./ N)); // type conver necessary?
    vector[2] recov = gamma .* I;
    
    // S states (lo, hi)
    dydt[1] = -(phi[1] *S[1]) + recov[1]; // dS/dt
    dydt[2] = -(phi[2] *S[2]) + recov[2];
    //dydt[1:2] = -(phi .* S) + recov;
    
    // I states
    dydt[3] = (phi[1]*S[1]) - recov[1]; // dI/dt
    dydt[4] = (phi[2]*S[2]) - recov[2];
    //dydt[3:4] = (phi .* S) - recov;
    
    return dydt;
  }
  
  vector incidR(vector S, vector I, matrix probM, vector c, real beta) {
    // Transpose from inbound row to col vecs????????
    vector[2] N = S + I;
    vector[2] phi = (beta * c) .* to_vector(probM * to_matrix(I ./ N));
    return phi .* S;
  }
  
  // Only need recov, cheaper calc
  vector indir_incidR(vector It1, vector It2, vector recov) {
    vector[2] incid;
    incid[1] = (It2[1] - It1[1]) + recov[1]; //lo
    incid[2] = (It2[2] - It1[2]) + recov[2]; //hi
    return incid;
  }
  
  vector recovR(vector I, real gamma){
    return gamma .* I;
  }
  
}

// Input data
data {
  int<lower=0> ntime;
  real<lower=0> new_cases[ntime]; // New Infs
  real<lower=0> pop_sus[ntime+1]; // Obs possible susp
  real<lower=0> ts[ntime];
  int<lower=0,upper=1> compute_loglik;
  matrix[2,2] probM;
}

transformed data {
  int<lower=0> ntime_w0;
  ntime_w0 = ntime + 1;
}


parameters {
  vector<lower=0,upper=5e6>[2] Nstate0; // Initial Sus and Inf; cant be more than 5 million suscep
  real<lower=0> s_sigma; // Overall var S
  real<lower=0> i_sigma; // Overall var I
  real<lower=0,upper=1> beta; // Trans prob
  real<lower=0> gamma; // Recovery rate
  real<lower=0, upper=1> frac; // fraction of high cont
  vector<lower=0>[2] c; // 1 low, 2 high
  real<lower=0> p_s; 
  real<lower=0,upper=1> p_i; // Surveillance prop
}

transformed parameters {
  array[ntime_w0] vector<lower=0>[4] y; // Sus and Inf states after 0 time
  array[ntime] vector<lower=0>[4] rates; // Sus and Inf states after 0 time
  
  // Initial conditions, from predicted totals...
  y[1,1] = frac * Nstate0[1]; // S0lo
  y[1,2] = (1- frac) * Nstate0[1]; // S0hi
  y[1,3] = frac * Nstate0[2]; // I0lo
  y[1,4] = (1- frac) * Nstate0[2]; // I0hi
  
  // time 1 is 0, for incid, need to extend so can all be in 1 array (initial time is 0 because first data is increase in cases)
  y[2:ntime_w0, 1:4] = ode_rk45(sis, y[1,1:4], 0, ts, probM, c, beta, gamma); 
  
  // One step less
  for(t in 1:ntime) {
    rates[t, 3:4] = recovR(y[t, 3:4], gamma); // ntime vector, backstep to keep 1:max
    rates[t, 1:2] = indir_incidR(y[t, 3:4], y[t+1, 3:4], rates[t, 3:4]); // Provide I across states and with recov rate
    //rates[t, 1:2] = incidR(y[t, 1:2], y[t, 3:4], probM, c, beta);
    
  }
  
}

model {
  // Priors
  Nstate0[1] ~ lognormal(log(8e5), 0.4);
  Nstate0[2] ~ lognormal(log(5e3), 1);
  s_sigma ~ exponential(1);
  i_sigma ~ exponential(1);
  beta ~ beta(10, 2.5); // Recall its dt per quarter
  gamma ~ normal(1.5, 1); // Weigh on more than a few days recovery (dt is quarter... ~1/0.33)
  frac ~ beta(3, 100); // Most likely under 10%
  c[1]~ normal(.25, 0.3); // Lo
  c[2]~ normal(5, 2); // Hi
  p_s ~ normal(3, 1); // Truncated normal 
  p_i ~ beta(40, 200);
  
  if(compute_loglik == 1) {
    
    // Likelihood (loop otherwise STAN doesnt know how to multiply)
    for (t in 1:ntime_w0) { 
      pop_sus[t] ~ lognormal(log(sum(y[t,1:2]) * p_s), s_sigma);
      
      // change in incid is 1 less in size than all suscep
      if (t < ntime_w0) {
        new_cases[t] ~ lognormal(log(sum(rates[t,1:2]) * p_i), i_sigma);
      }
    }
  }
  
  
}

generated quantities {
  //real R0 = beta / gamma; // Rough estimate with assumptions on suscep of pop
  //vector[ntime] Reff; // S * R0, where S is prop suscpep at given time
  real recov_time = 1 / gamma;
  
  array[ntime_w0] real<lower=0> y_pred_s;
  array[ntime] real<lower=0> y_pred_i;
  
  vector[ntime_w0] log_likS;
  vector[ntime] log_likI;
  vector[ntime+ntime_w0] log_lik;
  
  for (t in 1:ntime_w0) { 
    y_pred_s[t] = lognormal_rng(log(sum(y[t,1:2])* p_s), s_sigma);
    log_likS[t] = lognormal_lpdf(pop_sus[t] | log(sum(y[t,1:2])*p_s), s_sigma);
    if (t < ntime_w0) {
      y_pred_i[t] = lognormal_rng(log(sum(rates[t,1:2]) * p_i), i_sigma);
      log_likI[t] = lognormal_lpdf(new_cases[t] | log(sum(rates[t,1:2])*p_i), i_sigma);
    }
  }
  
  log_lik = append_row(log_likS, log_likI); // Combined loglik
  
}
