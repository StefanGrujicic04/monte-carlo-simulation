# monte-carlo-simulation
A Python project demonstrating Monte Carlo estimation and probability modeling using binomial simulations.
# Monte Carlo Simulation of a Binomial Process

Author: **Stefan Grujičić**  
Project: `Monte Carlo Simulation for Binomial Estimation`  
Language: Python (NumPy, Matplotlib, Pandas)

---

##  Project Overview

This project demonstrates how **Monte Carlo methods** can be used to estimate parameters and event probabilities in a **binomial process** — a common probabilistic model describing repeated independent events with two possible outcomes (e.g., rain/no rain, success/failure).

The goal is to **simulate**, **analyze**, and **compare** empirical results against theoretical predictions of the binomial distribution, showing that large-scale random sampling can accurately approximate statistical expectations.

---
## list of required Python packages

numpy
matplotlib
pandas

---
##  Concept and Motivation

Monte Carlo algorithms rely on *random sampling* to approximate mathematical quantities that may be difficult or impossible to compute analytically.

In this case, we model the number of “successes” (rainy days) in a fixed number of trials (7 days) with probability `p` of success on each day.

We estimate:

- The parameter `p` (via the Maximum Likelihood Estimator, MLE)
- The probability of a specific event `A = {X = 5}`
- 95% confidence intervals for both `p` and `P(A)`
- The convergence of estimates as the number of simulations increases

This illustrates how random simulation approaches theoretical statistical behavior when the sample size is large — a practical demonstration of the **Law of Large Numbers** and the **Central Limit Theorem**.

---

##  Implementation Details

### Model
We assume:
\[
X \sim \text{Binomial}(m=7, p)
\]
and define the event:
\[
A = \{ X = 5 \}
\]

### Simulation Steps
1. Generate `n` independent samples from `Binomial(7, p)` using NumPy.
2. Compute:
   - Mean value `X̄`
   - Estimated parameter `p̂ = X̄ / 7`
   - Empirical event probability `P̂(A) = P(X=5)`
3. Construct 95% confidence intervals for `p` and `P(A)` using CLT and the delta method.
4. Compare:
   - **True theoretical values** vs **simulated estimates**
   - **Empirical histogram** vs **theoretical binomial PMF**
5. Visualize convergence and distribution.

---

##  Usage

Run the simulation with:
```bash
python3 numericka.py --n=200000 --seed=42 --out=results
