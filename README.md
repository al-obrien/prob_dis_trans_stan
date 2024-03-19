
<!-- README.md is generated from README.Rmd. Please edit that file -->

# sis_stan

<!-- badges: start -->
<!-- badges: end -->

This project is to understand how to add compartmental models to STAN so
that deterministic and probabilistic model components can be combined,
all through an R interface. The model will be inherently flawed in its
simple state as it makes strong assumptions around homogeneity of mixing
for an STI, which is unequivocally untrue for gonorrhea. I could have
simulated some data to avoid this blasphemy, but real data is more fun
and interesting albeit more complex. Furthermore, although a **SEIRS**
model is likely more representative of an infection like gonorrhea, the
simplicity of the **SIS** model will be used for learning purposes.
Later models, if time permits, will address these limitation, perhaps by
adding more complex details with contact patterns and adding more
compartments.

# SIS Model

Assumes no incubation period, and becomes susceptible following
infection (no convalescent period). Furthermore, to start, there is no
vital effects and the model is a closed system.

## Compartments

$$
\begin{align*}
&S \rightarrow I \rightarrow S \\
&I \overset{\beta SI}{\underset{\gamma I}{\iff}} S 
\end{align*}
$$

## Rate of change

$$
\begin{align*}
\frac{dS}{dt} &= \gamma I - \frac{\beta SI}{N} \\
\frac{dI}{dt} &= \frac{\beta SI}{N} - \gamma I \\\\
\beta_{infection\_rate} &= contact * transmission  \\
\gamma_{recovery\_rate} &= 1/duration \\
0 &= \frac{d}{dt}(S + I) \\
S + I &= N = \text{constant} 
\end{align*}
$$

## Change in state over time

$$
\begin{align}
\text{Susceptible:}\\
S_{t+dt} &= S_t + dt(\frac{dS}{dt})\\
&= S_t - dt\beta S_tI_t + dt\gamma I_t\\ \\
\text{Infected:}\\
I_{t+dt} &= S_t + dt(\frac{dI}{dt}) \\
&= S_t + dt\beta S_tI_t - dt\gamma I_t
\end{align}
$$

# Probabalistic Model

The deterministic components have made some assumptions about how we
believe the disease is transmitted, but we need to incorporate
randomness to account for imperfect knowledge about this process. For
example, we only have a window into the Infected cases through
surveillance programs, which are passive. We want to use our limited
data and belief about the population in Alberta to fit this model with
the deterministic engine outlined above. Thankfully, Stan is able to
account for this. Although the log-normal is used, other count specific
distributions could also be used, such as the negative-binomial.

$$
\begin{align}
&\textbf{Likelihood}\\
&s_t \sim logNormal(log(qS_t), \sigma_t) \\
&i_t \sim logNormal(log(pI_t), \sigma_t) \\\\
&\textbf{Priors}\\
&\sigma \sim exponential(1) \\
&p \sim halfNormal() \\
&p \sim beta() \\
&\beta \sim halfNormal() \\
&\gamma \sim halfNormal() \\
\end{align}
$$

# References

- [Bayesian Workflow with diseases and
  STAN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8661657/)
- [Heterogeniety in Disease
  Transmission](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4808916/)
- [Dr. Tuite’s thesis on STI math
  models](https://tspace.library.utoronto.ca/bitstream/1807/71350/1/Tuite_Ashleigh_201511_PhD_thesis.pdf)
