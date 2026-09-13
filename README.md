# Bayesian Marketing Mix Modelling

A complete Bayesian MMM in **pure Python, standard library only**: geometric adstock, Hill saturation,
hand-derived analytic gradients, a **No-U-Turn Sampler** with dual averaging and a diagonal mass matrix,
rank-normalised split R-hat and effective sample size, posterior predictive checks, parameter recovery,
holdout forecasting, and budget allocation with the uncertainty attached. No PyMC, no Stan, no NumPy.

> The output blocks below are **schematic illustrations of the report format**, laid out by hand to show what
> each command prints and what to look at. They are not transcripts of a recorded run: the figures depend on
> the seed and the machine, so run the commands and read your own.

The interesting part of an MMM is not the fit. Two years of weekly sales against a dozen parameters will fit
whatever you ask it to. The interesting part is knowing when the answer is real, and this repository is built
around the failure that decides it.

```
$ python -m mmm identifiability

regime 'clean'                             regime 'flat'  (near-constant spend)
  search spend ranges 1,800 - 71,000         search spend ranges 21,000 - 22,900
  channel   true kappa   width / prior       channel   true kappa   width / prior
  search        12,000          ~0.2         search        12,000          ~0.9
  tv            30,000          ~0.3         tv            30,000          ~1.0
  social         9,000          ~0.2         social         9,000          ~0.9
  residual sd matches the true noise         residual sd matches the true noise
```

Both fits are excellent. Both will produce a confident budget recommendation. In the second one the
saturation posterior is **still the prior** -- the last column is posterior interval width over prior interval
width, and a ratio near 1 means the data taught the model nothing. Spend that never moves cannot trace out a
saturation curve, so the recommendation is prior belief wearing a posterior's clothes, and no goodness-of-fit
statistic will say so. The cure is spend variation, which means an experiment, not more weeks of the same plan.

```bash
python -m mmm gradient         # the analytic gradient against central finite differences
python -m mmm sampler          # NUTS on a Gaussian with a known answer, random walk as the control
python -m mmm fit              # sample, diagnose, recover, decompose, price the media
python -m mmm identifiability  # the quiet failure above
python -m mmm confounded       # the loud one: media takes credit for Christmas
python -m mmm budget           # allocation, verified against brute force, with decision intervals
python -m mmm forecast         # a 13-week holdout against same-week-last-year
```

---

## The model

```
y_t = base + trend * t/(T-1) + seasonality_t + sum_c beta_c * Hill(adstock(x_c)_t) + sum_k eta_k z_kt + e_t
e_t ~ Normal(0, sigma)
```

with Fourier seasonality (`sin`/`cos` at 1 and 2 cycles per year), control variables for price and holidays,
and per-channel media transformations.

**Carryover.** `A_t = x_t + lambda * A_{t-1}`, normalised by `(1 - lambda)`. The normalisation is not
cosmetic: without it total adstocked volume is `sum(x)/(1 - lambda)`, so `beta` must shrink as `lambda`
grows, and the posterior develops a banana-shaped `beta`-`lambda` ridge that a sampler has to fight. With it,
total volume equals total spend and the two parameters largely decouple.

**Saturation.** `s(u) = u^alpha / (u^alpha + kappa^alpha)`. Chosen because the parameters mean something:
`kappa` is the half-saturation point in spend units (`s(kappa) = 0.5` exactly) and `alpha` is the shape.
`alpha <= 1` is concave from the origin; `alpha > 1` is an S-curve with a convex toe -- a real claim about the
market, and one the budget optimiser has to respect.

### Sampling in an unconstrained space

Hamiltonian dynamics cannot respect a boundary, so every constrained parameter is sampled through a
bijection: `beta = exp(b)`, `kappa = exp(h)`, `alpha = exp(a)`, `lambda = sigmoid(l)`, `sigma = exp(s)`. Each
transformation contributes a log-Jacobian, and **omitting one is the classic silent MMM bug** -- the chains
still converge, just to the wrong posterior. Two of them simplify, and the simplification is documented
rather than assumed:

- A LogNormal(mu, tau) prior on `beta` plus `|dbeta/db| = beta` is exactly a Normal(mu, tau) prior on `b`;
  the `-log beta` in the density and the `+log beta` from the Jacobian cancel.
- A Beta(a, b) prior on `lambda` plus `|dlambda/dl| = lambda(1-lambda)` gives
  `a log lambda + b log(1-lambda) - log B(a,b)`, and the gradient collapses to `a(1-lambda) - b*lambda`.

### The gradient, by hand

With `r_t = y_t - mu_t`, the likelihood contributes `dLL/dmu_t = r_t/sigma^2`, and each parameter picks up its
own `dmu_t/dparam`:

| coordinate | derivative |
|---|---|
| `base`, `trend`, `season_j`, `control_k` | 1, `t/(T-1)`, the Fourier term, `z_kt` |
| `log beta_c` | `beta_c * s_ct` |
| `logit lambda_c` | `beta_c * ds/du * da_t/dlambda * lambda(1-lambda)` |
| `log kappa_c` | `beta_c * ds/dkappa * kappa_c` |
| `log alpha_c` | `beta_c * ds/dalpha * alpha_c` |
| `log sigma` | `sum_t (r_t^2/sigma^2 - 1)` |

where `da_t/dlambda` comes from differentiating the adstock recursion (`dA_t/dlambda = A_{t-1} + lambda *
dA_{t-1}/dlambda`) and the Hill partials are

```
ds/du     =  alpha * u^(alpha-1) * K / D^2
ds/dkappa = -alpha * kappa^(alpha-1) * U / D^2          U = u^alpha,  K = kappa^alpha,  D = U + K
ds/dalpha =  U * K * (ln u - ln kappa) / D^2
```

All of it is checked against central finite differences, coordinate by coordinate, at the true parameters and
at random points. That test is the foundation of everything downstream.

---

## The sampler

Written out in full: leapfrog integration, slice-sampled tree doubling with the no-U-turn criterion checked on
every subtree, Nesterov dual averaging for the step size, and a **diagonal mass matrix** estimated during
warmup. The metric is what makes this usable here -- `base` is in revenue units and `log alpha` is
dimensionless, and without a metric one step size must satisfy the tightest coordinate while the widest one
crawls.

**Divergences are counted and reported, never swallowed.** A divergence marks a region the integrator could
not enter, so that region is under-sampled and the fit is not trustworthy however healthy the traces look.

The sampler is validated against a target whose answer is known by algebra rather than against the model it
was written for: a two-dimensional Gaussian with correlation 0.95 and marginal scales 1 and 10, where the
recovered means, standard deviations and correlation can be compared with the values that were written down.
`python -m mmm sampler` prints that comparison, and runs a tuned random walk on the same evaluation budget as
a control -- it cannot see the ridge, so its effective sample size stays small however long it runs.
`tests/test_sampling.py` asserts the same recovery within stated tolerances, so the claim is checked rather
than displayed.

## Diagnostics

Rank-normalised split R-hat (Vehtari et al., 2021), effective sample size via Geyer's initial positive
sequence, and Monte Carlo standard error, all from scratch. Split-halving catches the failure that matters in
practice: a chain sliding slowly down a ridge looks perfectly converged to any between-chain comparison, and
`tests/test_sampling.py` includes exactly that case -- two identical drifting chains where the unsplit
statistic reads about 1.00 and the split statistic does not.

## From posterior to decision

**Average ROAS answers a question nobody asked.** Contribution over spend is a report on money already
committed; the decision -- move the next pound -- depends on the marginal return where you are standing. On a
saturated channel the two differ by a factor of several and can rank channels in opposite orders. Both are
reported side by side, and the inequality is tested in both directions: marginal below average on a concave
channel, above it inside the S-curve toe.

**Carryover leaks past the window.** Spend in the final week is still selling after the data ends, so
contributions are computed over an extended horizon and the tail share is reported rather than dropped.

**Allocation is solved at a defined operating point.** For constant weekly spend `s`, normalised adstock
converges to exactly `s`, so the steady-state problem is

```
maximise  sum_c beta_c * Hill(s_c)    subject to   sum_c s_c = B,  s_c >= 0
```

Concave channels: water-filling by bisection on the shadow price, which equalises marginal returns. With any
S-curve the problem is **not concave** -- funding a channel a little can be worse than not funding it at all
-- so the funded subset is enumerated and the concave problem solved within each. The result is checked
against a brute-force simplex grid in the tests, because an optimiser that quietly returns a stationary point
is worse than none.

And the output is an interval, not a number: `posterior_allocation` re-solves the allocation under every
posterior draw and reports the 5th, 50th and 95th percentile share for each channel. A median split with each
share uncertain to fifteen points is not a plan; it is a case for an experiment. Saying so is the difference
between a model that informs decisions and one that launders them.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q                    # the full suite
pytest -q -m "not slow"      # skip the one end-to-end fit
```

| file | what it pins down |
|---|---|
| `test_transforms.py` | hand-computed adstock and Hill values; every partial against finite differences |
| `test_model.py` | the gradient in every parameter block; that the generator and the model agree exactly |
| `test_sampling.py` | NUTS against a known Gaussian; metric adaptation; R-hat and ESS on constructed sequences |
| `test_decision.py` | the ROAS inequalities; the optimiser against brute force on non-concave parameters |
| `test_validate.py` | predictive checks that must fire, recovery coverage, chronological splitting |

The finite-difference tolerance is 1e-4 relative, and the reason is written into the test module: with a
log-posterior of order 1e3, cancellation floors any finite-difference estimate near 1e-8, so coordinates with
small gradients cannot be verified more tightly however correct the analysis is. Every error the check exists
to catch -- a missing Jacobian, a dropped chain-rule factor, a sign -- appears at order 0.1.

## Limits

- **Pure Python.** A 104-week, three-channel fit with 300 warmup and 300 draws on two chains takes minutes,
  not seconds. This is a study of the mathematics, not a production sampler.
- **No hierarchy.** One geography, one product. The multi-region partial-pooling model that makes real MMMs
  identifiable -- borrowing strength across markets so each one need not trace its own saturation curve -- is
  not implemented, and it is the single most valuable thing that could be added.
- **Geometric carryover only in the fitted model.** Real television peaks a week or two after exposure;
  `simulate_delayed` generates exactly that and the model cannot represent it. There is no fix here, only a
  demonstration of what it costs: ROAS survives, campaign timing does not.
- **Normal likelihood, additive channels.** No interaction terms, no multiplicative baseline, no
  heteroscedasticity, and no lift-test priors calibrating the media effects -- which is how the
  identifiability problem is actually solved in industry.
- **Allocation is steady-state.** Flighting -- when to burst and when to go dark -- is a different and harder
  problem than how much to spend per week.
- **Synthetic data only.** No proprietary dataset is used or implied.

## References

- Jin, Wang, Sun, Chan & Koehler (2017), *Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects* -- the adstock and Hill formulation used here.
- Chan & Perry (2017), *Challenges and Opportunities in Media Mix Modeling* -- on why observational MMM estimates are fragile.
- Hoffman & Gelman (2014), *The No-U-Turn Sampler* -- tree doubling and dual averaging.
- Betancourt (2017), *A Conceptual Introduction to Hamiltonian Monte Carlo* -- on divergences and what they mean.
- Vehtari, Gelman, Simpson, Carpenter & Burkner (2021), *Rank-Normalization, Folding, and Localization: An Improved R-hat*.
- Geyer (1992), *Practical Markov Chain Monte Carlo* -- the initial positive sequence estimator.

MIT licensed.
