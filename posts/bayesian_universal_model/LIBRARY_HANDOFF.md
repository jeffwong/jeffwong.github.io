# Handoff: from blog research to a production library

**Universal model, compression, and Bayesian bootstrap**

*Load this file first in a new repository and session. It is self-contained: it gives you the math, the algorithms, the non-obvious engineering decisions behind the single-pass kernels, the tests that prove correctness, and how the user likes to work. Written 2026-10-18, at the end of the research phase.*

---

## 0. How to load this context (read in this order)

You are starting a new repository whose goal is a production-grade software library. The research behind it lives in a separate repository, Jeffrey Wong's Quarto blog at `/Users/jeff/coding/jeffwong.github.io`. Your memory directory for the new repo will be empty: nothing from the blog sessions carries over except this file. Read:

1. **This file, completely.** Sections 3–6 are the spec. Section 7 is the test oracle.
2. `/Users/jeff/coding/jeffwong.github.io/posts/conditionally_sufficient_statistics/index.qmd` (the "CSS post"). It is the universal model and its compression. Its code chunks are the reference implementation for point estimates, covariances and effects. Key chunk labels:
   - `world`: the simulator
   - `covariates`: `prepare` and `centered`
   - `design-matrices`, `model`: `fit_rows`, `complier_mean`
   - `effects`: delta vectors and complex-step gradients
   - `designs`: 16-design check against estimatr
   - `compression-functions`: `compress_summaries`, `fit_first_moments`, `L_matrix`, `stack_instruments`, `fit_compressed`, `complier_mean_compressed`
   - `kernel`: Rcpp `account_summaries`, `compress_profiles`
   - `general-compressed`: `distinct_index`, `compress_rows`
   - `early`: delta vectors early, `projection`
   - `second-order`: approximation for many covariates
   - `scores`: prognostic scores
3. `/Users/jeff/coding/jeffwong.github.io/posts/bayesian_universal_model/index.qmd` (the "BUM post"). It is the Bayesian bootstrap of the universal model. Key chunk labels:
   - `draw-rows`: the reference posterior on rows
   - `fit-sums`: effects from stratum sums
   - `kernel`: Rcpp `bootstrap_weights`, `compress_bootstrap`, `compress_moments`
   - `one-pass`: `strata`, `posterior_one_pass`, `posterior_from_sums`
   - `gaussian`: `posterior_compressed`
   - `mc`: calibration Monte Carlo
4. `/Users/jeff/coding/jeffwong.github.io/posts/bootstrap/index.qmd` covers the classical, Poisson and Bayesian bootstraps as reweightings. It derives the one-pass weighted least squares computation and shows that the linearized bootstrap is the HC0 sandwich. Read it for background only.
5. Optional, for older research scripts: local git branches in the blog repo, `snapshot/cond-suff-stats-milestone-{1..4}`, under `research/conditionally_sufficient_statistics_v*/`. They are not pushed. Read them with, for example, `git -C /Users/jeff/coding/jeffwong.github.io show snapshot/cond-suff-stats-milestone-2:research/conditionally_sufficient_statistics_v2/univ.R`. The BUM research scripts lived in a session scratchpad that no longer exists. Everything they verified is recorded in Section 7 and reproduced by the BUM post's chunks.

**First actions in the new session:**
1. Write a `CLAUDE.md` for the new repo. Section 9 says what goes in it.
2. Confirm the open decisions in Section 10 with the user (`AskUserQuestion`). Do not pick a language or framework silently.
3. Port the reference implementation to the target language and make the oracle tests in Section 7 pass before optimizing anything.

---

## 1. What the library does, in one paragraph

The library fits one model that covers most online experiment designs: **weighted two stage least squares, with every covariate interacted with treatment, cluster robust standard errors, and multiple outcome columns**. Switching features on or off gives the textbook analysis of each design:
- AB test with Lin regression
- cluster randomized test
- encouragement design (LATE)
- staggered or ramped propensities (IPW)
- ratio metrics

The library computes effects (the ATE or LATE, effects for a segment, slopes, ratio metrics, relative effects) and their uncertainty in three forms:
1. sandwich standard errors via delta vectors;
2. exact Bayesian bootstrap posteriors;
3. approximate posteriors drawn from the compressed data.

It does all of this from a **compressed dataset** built in **one pass** over the raw rows. The compressed dataset has one row per stratum, its size does not depend on the number of rows, and it is mergeable, so it works with map-reduce. Production scale is hundreds of millions of rows and dozens of metrics.

---

## 2. Vocabulary and notation (keep it; the user is strict about it)

The user requires one word per data level and one symbol per meaning. Code names must mirror the math.

| Term / symbol | Meaning |
|---|---|
| **profile row** | One raw row (unit of analysis, e.g. a profile `p` of account `a`). |
| **account** $a$ | The **cluster**: the unit of randomization *and* of independence. Every bootstrap weight and every cluster-robust sum is per account. |
| **account summary** $u_a$ | Vector of sums over the account's profile rows. It is an intermediate: built in a buffer, consumed, never stored. |
| **stratum** $h = (Z, \text{day})$ | Assignment arm × the propensity cell (cohort). All accounts in a stratum share $Z_a$ and $w_a$. |
| **compressed row** | Per stratum: count $G_h$, first moments $\sum_{a\in h} u_a$, second moments $\sum_{a\in h} u_a u_a^\top$. |
| **compressed dataset** | All compressed rows (8 in the running example). |
| $Z_a$ | Randomized assignment (prompt / instrument), constant within account. |
| $T_{ap}$ | Treatment received, chosen by the profile. |
| $e_{\text{day}}$, $\hat e_{\text{day}}$ | Propensity of $Z=1$ in the cohort; the estimate is the **account-level** share of prompted accounts in that cohort. |
| $w_a$ | Inverse propensity weight: $1/\hat e$ if $Z_a=1$, $1/(1-\hat e)$ if not. $w=1$ when there is no ramp. |
| $x_{ap}$ | $k$ covariates, **centered at a fixed reference point $x_{ref}$ chosen before the data** (Section 5.6). |
| $d = k+1$ | Intercept plus covariates. |
| $o$ | Number of outcome columns. |
| $M_1$, $m_{1,ap}$ | Instruments $(1, x, Z, Zx)$. |
| $M_2$, $m_{2,ap}$ | Regressors $(1, x, T, Tx)$. Never call the design matrix $Z$: $Z$ is the instrument. |
| $\hat\beta$ | $2d \times o$ coefficient matrix (code: `beta`; `B` is reserved for the number of draws). |
| $\bar x_C$ | Compliers' mean covariates. |
| $\delta$ | Delta vector: reads a treatment line at a covariate point. |
| $\xi_a^{(b)}$ | Bayesian bootstrap weight of account $a$ in draw $b$, iid Exponential(1). |
| $\pi_a$ | Normalized shares $\xi_a/\sum\xi$. They are never needed in code (Section 4.2). |
| $v_a = (1, u_a)$ | Account summary with a leading 1. |
| $B$ | Number of posterior draws. |
| $q$ | Length of $u_a$. |

Outcome columns for the running example: $y$ (hours), $N$ (rebuffers), $D$ (plays), $TN = T\cdot N$, $TD = T\cdot D$. **$TN$ and $TD$ are required for ratio metrics under encouragement** (Section 3.3).

---

## 3. The universal model (math spec)

### 3.1 Both stages

With centered covariates:

$$
\begin{aligned}
y_{ap} &= \beta_{00} + x_{ap}^\top\beta_{0x} + T_{ap}(\beta_{10} + x_{ap}^\top\beta_{1x}) + \varepsilon_{ap} &&\text{second stage}\\
T_{ap} &= \gamma_{00} + x_{ap}^\top\gamma_{0x} + Z_a(\gamma_{10} + x_{ap}^\top\gamma_{1x}) + \nu_{ap} &&\text{first stage for } T\\
T_{ap}x_{apj} &= \gamma^{(j)}_{00} + x_{ap}^\top\gamma^{(j)}_{0x} + Z_a(\gamma^{(j)}_{10} + x_{ap}^\top\gamma^{(j)}_{1x}) + \nu^{(j)}_{ap} &&\text{first stage for interaction } j=1..k
\end{aligned}
$$

The model is just identified, with $\dim m_1 = \dim m_2 = 2d$, so:

$$\hat\beta = (M_1^\top W M_2)^{-1} M_1^\top W Y,\qquad W=\text{diag}(w_a \text{ for each profile row}).$$

Residuals use the *received* treatment: $\hat\varepsilon = Y - M_2\hat\beta$. The cluster robust (CR0) covariance of $\text{vec}(\hat\beta)$ is:

$$(I_o\otimes A^{-1})\Big[\sum_a s_a s_a^\top\Big](I_o\otimes A^{-1})^\top,\quad A = M_1^\top W M_2,\quad s_a = \text{vec}\Big(\sum_p w_a m_{1,ap}\hat\varepsilon_{ap}^\top\Big).$$

The CSS post uses CR0 and HC0, with no small-sample correction. It shows that the CR1 correction $\frac{G}{G-1}\frac{n-1}{n-\dim\beta}$ equals HC1 when $G=n$.

### 3.2 Reductions (all proven in the CSS appendix and tested against estimatr in 16 designs)

- **No encouragement** ($T=Z$): $M_1=M_2$, so the model is weighted least squares. That is Lin's regression with robust standard errors, and $\bar x_C$ is the weighted mean of $x$.
- **No clusters** (one profile per account): the cluster-robust covariance is HC0.
- **No ramp**: constant $w$, which cancels.
- **No ratio**: $D\equiv 1$, and $\tau_R$ reduces to the LATE on $N$.
- **No covariates**: the Wald ratio, or the difference in means.

The switches touch different parts of the model (instruments, the clustering of the meat, weights, outcome columns), so every combination is also covered.

### 3.3 Effects (always as functions of $\hat\beta$ plus delta vectors; never hand-pick coefficients)

- $\delta = (0_d,\,1,\,\bar x_C - x_{ref})$. Write $\ell_Y = \delta^\top\hat\beta_Y$ for the treatment line of outcome $Y$ read at $\bar x_C$.
- $\tau_{LATE} = \ell_y$. **The line must be read at the compliers' mean $\bar x_C$, not at the population mean.** The treatment line is a $\pi_C(x)f(x)$-weighted fit to $\tau_C(x)$. Reading it at $\bar x$ answers a different question.
- $\bar x_C$ is a Wald ratio of intent-to-treat effects: $\bar x_C = \dfrac{E[Tx|Z=1]-E[Tx|Z=0]}{E[T|Z=1]-E[T|Z=0]}$. Each arm's average is $\sum w T(1,x)/\sum w$ over its profile rows.
- Ratio metric: $\tau_R = \dfrac{\ell_{TN}}{\ell_{TD}} - \dfrac{\ell_{TN}-\ell_N}{\ell_{TD}-\ell_D}$.
  - $\ell_{TN} = E[N(1)|C]$ and $\ell_{TN}-\ell_N = E[N(0)|C]$.
  - The baseline line does **not** give the compliers' levels. Using it biased $\tau_R$ (in the CSS Monte Carlo the bias does not shrink with $n$).
- Relative ratio: $\rho_R = \dfrac{\ell_{TN}/\ell_{TD}}{(\ell_{TN}-\ell_N)/(\ell_{TD}-\ell_D)} - 1$.
- Segment effects read the line at another $x_0$. Slopes are the entries of $\beta_{1x}$.
- **Standard errors:** $\sqrt{g^\top V g}$ with $g$ the gradient of the effect with respect to $\text{vec}(\hat\beta)$.
  - The reference computes $g$ by **complex-step differentiation**: $\text{Im}\,f(\theta + i\kappa e_j)/\kappa$ with $\kappa = 10^{-20}$, which is exact to machine precision.
  - A library may use autodiff or analytic gradients instead. Test them against complex step.
  - Caveat: the CSS delta method holds $\bar x_C$ and $\hat e$ fixed. The Bayesian bootstrap does not. The difference was about 0.5–1% of the SD on average (Section 7).

### 3.4 Design assumptions

- **(A1)** Accounts are independent; profiles within an account may be correlated in any way.
- **(A2)** Within a cohort, $Z$ is randomized with probability $e_{\text{day}}$.
- **(A3)** Exclusion and monotonicity.

The bootstrap adds:
- **(A4)** The population contains only the kinds of accounts we observed: $F=\sum_a\pi_aF_a$.
- **(A5)** A Dirichlet prior with zero pseudo-counts, so the posterior is Dirichlet(1,…,1).

---

## 4. Bayesian bootstrap of the universal model (math spec)

### 4.1 The weight goes on the account

Each account gets one $\xi_a\sim\text{Exp}(1)$, shared by all its profile rows. **Per-profile weights are wrong.** They gave LATE SE/SD 0.856 and 90.9% coverage over 1000 Monte Carlo replications, against about 0.97 and 94% per account. This is the posterior analogue of using HC instead of CR.

### 4.2 One draw = the universal model refit with combined weight $\xi_a w_a(\xi)$, recomputing everything that depends on the population

1. Propensity: $e_{\text{day}}(\xi) = \sum_{a\in\text{day}}\xi_aZ_a / \sum_{a\in\text{day}}\xi_a$. Then $w_a(\xi)$ follows.
2. $\beta(\xi) = (\sum_a \xi_a w_a \sum_p m_1m_2^\top)^{-1}\sum_a\xi_aw_a\sum_p m_1y^\top$.
3. $\bar x_C(\xi)$: the Wald ratio with weights $\xi_aw_a$.
4. Effects: the same functions as in Section 3.3.

**Scale invariance:** every step is invariant to multiplying all $\xi$ by a constant, so **use the unnormalized $\xi$**. The Dirichlet normalization is never computed. With $\xi\equiv1$ the draw reproduces the point estimate exactly; this is a mandatory test.

The posterior of any function of the effects is that function applied to each draw. The library should accept user-supplied effect functions of $\hat\beta$ and $\bar x_C$.

### 4.3 Why it agrees with the sandwich

To first order, $\tau^{(b)}-\hat\tau\approx\sum_a(\xi_a-1)\psi_a$ with independent weights of variance 1. So the posterior variance is approximately $\sum_a\psi_a^2$, the same form as the cluster sandwich (Taddy et al. 2016, arXiv 1412.8563).

---

## 5. The efficient algorithms: the non-obvious parts

This is the part that is not obvious from the posts. Read it carefully before writing any kernel.

### 5.1 What every computation needs: per-account summaries, then per-stratum sums

Because $Z_a$ and $w_a$ are constant within an account, everything the fit needs reduces to three per-account blocks, with $R_{ap}=(1,x_{ap})$ the length-$d$ row:

$$S^Y_a=\sum_pR_{ap}^\top y_{ap}\ (d\times o),\quad S_a=\sum_pR_{ap}^\top R_{ap}\ (d\times d),\quad S^T_a=\sum_pT_{ap}R_{ap}^\top R_{ap}\ (d\times d).$$

$u_a = (\text{vec}S^Y_a,\ \text{vec}S_a,\ \text{vec}S^T_a)$, read column-major.
- Full layout: $q_{full} = do + 2d^2$ (102 in the example).
- Distinct layout: the upper triangles of the symmetric blocks, $q = do + d(d+1)$ (72). CSS `distinct_index(d, o)` maps distinct entries to full ones.

**Assembling the fit from sums.** With $s = \sum_h w_h U_h$ over all strata and $s_Z = \sum_h w_hZ_hU_h$ over prompted strata, where $U_h$ are the stratum sums of $u_a$:

$$M_1^\top WM_2 = \begin{pmatrix}S(s) & S^T(s)\\ S(s_Z) & S^T(s_Z)\end{pmatrix},\qquad M_1^\top WY = \begin{pmatrix}S^Y(s)\\ S^Y(s_Z)\end{pmatrix}.$$

**Compliers' mean from sums.** For each arm, take the column $S^T[\,\cdot,1]$ (the sums of $T\cdot(1,x)$) divided by $S[1,1]$ (the weighted profile count). Difference the arms and divide entries $2..d$ by entry 1. Without encouragement, use $S[2..d,1]/S[1,1]$ of all strata. BUM `fit_sums` implements exactly this; port it verbatim.

### 5.2 Strata must equal arm × the estimator's propensity cells, exactly

A compressed row may only merge accounts that share $Z_a$ *and* $w_a$. A past bug kept finer cells with count weights and got **86% coverage**. Merging two compressed rows of the same stratum is exact: add the fields. That makes the compressed row a commutative monoid, which is the property map-reduce needs.

**$T$ is not a grouping key.** It varies within an account, so it lives inside $u_a$ as $S^T_a$. Under encouragement, $T$ is an outcome-like column.

**Categorical covariates.** A categorical can be a grouping key with no one-hot encoding only if it is constant within the summed unit (the account). A profile-level categorical must become per-level indicator columns inside $u_a$.

### 5.3 Single pass, cluster-aware: the buffer-and-flush pattern

- Input rows must be **grouped contiguously by account**: sorted, or shuffled by account key in a distributed engine.
- Keep one $u$ buffer. For each row, add its products into the buffer.
- When the account id changes, **flush**: fold $u_a$ into its stratum's state, then zero the buffer.
- Zero only the touched entries. The kernels track the nonzero indices `nz` and skip zeros: for example, $S^T_a\equiv0$ for unprompted accounts, and device indicators are mostly zero.
- Account summaries are **never stored**: at 20k accounts they would hold more numbers than the raw rows.

The flush is the only place the algorithms differ:

| Algorithm | Stratum state | Flush cost per account | Output size |
|---|---|---|---|
| Sandwich, "delta vectors at end" (CSS `compress_profiles`) | $G_h$, $\sum u_a$, lower triangle of $\sum u_au_a^\top$ | $O(\text{nnz}^2/2)$ | $H(1+q+q(q+1)/2)$ |
| Sandwich, "delta vectors early" (CSS `early`) | pass 1: $\sum u_a$; pass 2: $\sum_a\psi_a^2$ per effect | pass 1 $O(\text{nnz})$; pass 2 $O(q)$ per effect | $Hq$ + one number per effect |
| Exact Bayesian bootstrap (BUM `compress_bootstrap`) | $B\times(1+q)$ matrix: column 0 $\sum\xi^{(b)}$, then $\sum\xi^{(b)}u_a$ | $O(B\cdot\text{nnz})$ | $HB(1+q)$ |
| Compressed-data bootstrap (BUM `compress_moments`) | $\sum v_a$, $\sum v_av_a^\top$ with $v=(1,u)$; **identical information to the sandwich state** | $O(\text{nnz}^2/2)$ | $H(q+1)(q+2)/2$ |

### 5.4 Delta vectors early (two passes, smallest storage)

The variance of an effect with gradient $g$ is $\sum_a\psi_a^2$, with:
- $\psi_a = c_h^\top u_a$;
- $c_h = w_h\,L(\hat\beta)^\top P_{Z_h}^\top (I_o\otimes A^{-\top})\,g$;
- $L(\hat\beta) = [\,I_{do},\ -(\hat\beta_0^\top\otimes I_d),\ -(\hat\beta_1^\top\otimes I_d)\,]$, which maps $u_a$ to $\text{vec}(E_a)$ with $E_a=S^Y_a-S_a\hat\beta_0-S^T_a\hat\beta_1$;
- $P_Z = I_o\otimes\begin{pmatrix}I_d\\ ZI_d\end{pmatrix}$.

The two passes:
- **Pass 1:** first moments, then $\hat\beta$, $\bar x_C$, the gradients, and $c_h$.
- **Pass 2:** rebuild each $u_a$, accumulate $(c_h^\top u_a)^2$.

The effects must be chosen before pass 2. CSS `L_matrix`, `stack_instruments` and `projection` are the reference.

### 5.5 The two Bayesian bootstrap engines

**(a) Exact one pass.**
- At the flush, for $b=1..B$: $\text{state}_h[b,0] \mathrel{+}= \xi^{(b)}_a$ and $\text{state}_h[b,k+1] \mathrel{+}= \xi^{(b)}_a u_k$.
- Weights come from a **counter-based hash of (seed, account id, b)**, mapped to $U\in(0,1)$ and then to $-\log U$.
  - Every shard computes the same weight for an account with no coordination, and nothing needs to know $n$ in advance.
  - The BUM kernel uses a splitmix64-style mixer on `seed + golden * (a*1000003 + b + 1)`. That is fine for research. **For production use a vetted counter-based RNG** (Philox or Threefry from Random123, or equivalent) and document the stream layout. The research hash's key packing collides if $b \ge 1000003$.
- **Memory layout matters (measured ~2× speedup):**
  - Store each stratum's state with column $k$ contiguous over $b$.
  - At flush, first compute the account's $B$ weights into a vector, then for each nonzero $k$ run a contiguous `axpy`: `col_k[0..B) += xi[0..B) * u[k]`.
  - The naive loop over $b$ outer and $k$ inner strides through memory.
  - The hash and log cost $O(B)$ per account. It is cheap relative to $B\cdot\text{nnz}$.
- **Per-draw fit:** take row $b$ of every stratum's state and call `fit_sums`. These $B$ small $2d\times2d$ solves are independent, so they parallelize trivially.
  - In R the overhead is assembling the $H\times(1+q)$ matrix. The fix was `cbind` of all strata once, then a `matrix(row, byrow=TRUE)` per draw.
  - In a compiled library, batch the solves.

**(b) From the compressed dataset (no extra pass).** This is the key identity, derived in the BUM post. The weights are independent, with mean 1 and variance 1, so given the data:

$$E\Big[\sum_{a\in h}\xi_av_a\Big]=\sum_{a\in h}v_a,\qquad \text{Cov}\Big(\sum_{a\in h}\xi_av_a\Big)=\sum_{a\in h}v_av_a^\top.$$

- **The compressed row *is* the posterior's first two moments.** The covariance is the **uncentered** second moment, because the randomness is in the weights, not in the accounts.
- Strata are independent.
- Draw each stratum's sum from $N(\text{first}, \text{second})$ and push the draw through the exact nonlinear `fit_sums`. That fit recomputes the propensity from entry 0, plus $\hat\beta$, $\bar x_C$ and the ratios.
- Use a **symmetric eigendecomposition with negative eigenvalues clamped to 0**, or a pivoted Cholesky. Plain Cholesky fails because:
  - the full layout duplicates the symmetric off-diagonal entries;
  - unprompted strata have $S^T\equiv0$;

  both make the covariance singular.
- **Equivalent reading:** this is *exactly* a multiplier bootstrap with N(1,1) weights. It matches the Bayesian bootstrap's first two weight moments, but negative weights mean a draw is not a population.
- **What it loses:** the third moments of the weighted sums. The Exp(1) weights have third central moment 2.
  - In the example the posterior SDs agree within Monte Carlo noise: 0.0581 vs 0.0587 for the LATE at 10k draws.
  - The skewness of $\rho_R$ was 0.29 exact vs 0.18 compressed. Most of the skew comes from the nonlinear map, which (b) keeps.
  - Use (a) for small strata or heavy-tailed accounts ("whales"). Taddy et al. saw skewed posteriors even at 13M users.
  - The smallest stratum in the example has 412 accounts.
- **Possible upgrade (not researched):** keep per-stratum third moments, or draw $\sum\xi$ exactly as Gamma($G_h$) and condition. Flag this to the user before building it.

**Storage crossover.** (a) grows like $B k^2$ and (b) like $k^4$. They cross near $q\approx 2B$, which is $k\approx40$ for $B=1000$ and $o=5$.

| $k$ | Numbers in raw rows (20k accounts) | (a) exact, $B=1000$ | (b) compressed |
|---|---|---|---|
| 3 | 439,890 | 328,000 | 6,888 |
| 5 | 527,868 | 584,000 | 21,608 |
| 10 | 747,813 | 1,504,000 | 142,128 |
| 25 | 1,407,648 | 6,664,000 | 2,778,888 |
| 50 | 2,507,373 | 23,264,000 | 33,837,488 |

**Default to (b).** It reuses the dataset the platform already stores for standard errors, and it supports any $B$ and any effect after the fact. Use (a) when strata are small or heavy-tailed, or when exactness is required. (a) needs $B$ and the seed fixed before the pass.

### 5.6 Numerical precision

- **Center covariates at a fixed $x_{ref}$ chosen before the pass**, not at the sample mean, which is unknown in one pass. The second moments contain fourth powers of $x$, and raw fourth moments lose precision.
- Any $x_{ref}$ gives the same fitted model; it only changes what $\beta_{00}$ and $\beta_{10}$ describe.
- Consider compensated summation, or float64 accumulators with per-shard partial sums.
- The reference implementation never needed more than float64 at 1M accounts.

### 5.7 Many covariates (from the CSS post; keeps storage bounded)

- **Prognostic scores (CUPAC):** replace $k$ covariates with one pre-experiment prediction per raw outcome.
  - They are as precise as all 25 covariates, and storage does not depend on $k$.
  - Score uncertainty does not need to enter the standard error, provided the score is trained on pre-experiment data or cross-fit.
- **Second-order approximation:** keep only the account totals of $(y, R, TR)$, which is $o+2d$ numbers per account, for the second moments.
  - It is valid only for effects read at $\bar x_C$ (LATE, ratio). Not for segment effects or slopes.
  - Leverage $1+(x_0-\bar x_C)^\top\Sigma_C^{-1}(x-\bar x_C)$ is constant at $x_0=\bar x_C$.
- Binning or k-means without regard to the outcome loses precision and saves little when there are several profiles per account.

---

## 6. How the research code was derived (the method to keep using)

The research followed the same pipeline each time. Keep it for the library: it is how correctness was established.

1. **Simulator with known truth.**
   - `simulate_accounts` (CSS `world` chunk) generates accounts × profiles with cohorts, a ramped propensity, an unobserved engagement that drives both take-up and outcome (so naive comparisons are confounded), shared account quality (so clusters matter), and Poisson ratio components.
   - The **truth** comes from simulating the same seed twice, once with every account prompted and once with none, and comparing compliers across the two worlds (`truth_pair`, 2×1M accounts).
   - Switches: `full_compliance`, `one_profile`, constant `propensity`.
2. **Textbook oracle.** Before writing anything clever, fit each design with estimatr: `iv_robust` or `lm_robust`, with `weights`, `clusters` and `se_type = "CR0"` or `"HC0"`. Center the covariates at $\bar x_C$ so the coefficient on $T$ is the effect.
3. **From-scratch matrix implementation** (`fit_rows`) that matches the oracle to floating-point rounding in all 16 designs.
4. **Derive the sufficient statistics** by asking which sums the estimator touches (Section 5.1). Then write the base cases by hand: one continuous covariate (the CUPED aggregates), and one categorical covariate (counts by level). Then the general kernel.
5. **Rcpp kernel** for the one-pass loop, checked against the R path to machine precision (bit-identical in BUM).
6. **Monte Carlo calibration**: at least 1000 replications. Report bias with its Monte Carlo SE, SE/SD, and coverage. Before calling anything biased:
   - (i) rule out Monte Carlo noise: SE/SD noise is about $1/\sqrt{2(R-1)}$ and coverage noise about $\sqrt{.95\cdot.05/R}$;
   - (ii) check the estimand, i.e. what the estimator converges to;
   - (iii) check that the delta vector is evaluated at the right covariate point.
7. **Profile before optimizing** (Rprof). Measured wins:
   - the cache-friendly bootstrap layout;
   - batching draw assembly;
   - no estimatr inside Monte Carlo loops;
   - `rowsum` or Rcpp instead of dplyr `group_by` on millions of rows;
   - `gc()` and dropping big objects before forking.

---

## 7. Test oracle: verified results the library must reproduce

### Invariants (language-independent; make these unit and property tests)

1. **Compressed equals rows.** For every one of the 16 designs, effects and covariance from the compressed dataset equal the row-level fit to floating-point rounding. Effects and SEs equal estimatr `iv_robust` / `lm_robust` with the same weights, clusters and CR0/HC0.
2. **Merge.** Compressing shards separately and adding the states equals compressing all rows at once. Test shard boundaries that split an account: must fail loudly or be handled by a pre-shuffle.
3. **Identity draw.** A bootstrap draw with all $\xi=1$ reproduces the point estimates exactly.
4. **Exact one pass equals rows.** For the same weights, the kernel's per-draw effects equal the row-level refit (`draw_rows`) to ~1e-13.
5. **Hash determinism.** `bootstrap_weights(account, B, seed)` is identical across shards and runs. The weights have mean ≈1, variance ≈1 and third central moment ≈2.
6. **Moments identity.** Over many weight draws, the empirical mean and covariance of $\sum_{a\in h}\xi_av_a$ converge to (`first`, `second`).
7. **Linearization.** Posterior SD ≈ sandwich SE within Monte Carlo noise in all 8 designs (BUM `designs` table).
8. **Delta vectors early equal at the end.** Same SEs to floating-point rounding.

### Regression numbers (R, seed `set.seed(2026)`, 20,000 accounts, 43,989 profile rows; reproducible only with R's RNG)

| Quantity | Value |
|---|---|
| truth $\tau_{LATE}$, $\tau_R$, $\rho_R$ | 0.6133, −0.02725, −0.3701 |
| point estimate LATE / CSS SE | 0.6639 / 0.0572 |
| point estimate $\tau_R$ / CSS SE | −0.02548 / 0.00410 |
| point estimate $\rho_R$ | −0.3378 |
| BB posterior SD (1000 draws, set.seed(2)) LATE / $\tau_R$ / $\rho_R$ | 0.0583 / 0.00424 / 0.0448 |
| BB posterior SD with $\bar x_C$ held fixed, LATE | 0.0566 |
| BB per-profile weights, LATE SD | 0.0535 (8% too narrow) |
| exact vs compressed posterior SD, LATE, 10k draws | 0.0581 vs 0.0587 |
| strata accounts $G_h$ | 7639, 5321, 2988, 1014, 412, 634, 981, 1011 |

### Calibration (1000 Monte Carlo replications, 500 draws each)

| Method | LATE SE/SD | LATE coverage | $\tau_R$ coverage |
|---|---|---|---|
| CSS sandwich | 0.963 | 94.5% | 94.4% |
| BB exact, per account | 0.968 | 94.0% | 94.4% |
| BB compressed (Gaussian) | 0.975 | 93.7% | 94.8% |
| BB per profile (wrong) | 0.856 | 90.9% | 94.6% |

The posterior mean of $\rho_R$ has bias 0.002: 1.6 Monte Carlo SEs and 5% of an SD, the $O(1/n)$ bias of a ratio.

### Performance reference (Apple Silicon laptop, 1M accounts, 2.19M profile rows, $k=5$, $o=5$, $B=1000$)

| Step | Seconds |
|---|---|
| Refit on rows, 1000 draws (extrapolated) | 684 |
| Exact one pass (Rcpp) | 11.7 |
| `compress_moments` + 1000 Gaussian draws | 1.2 |
| CSS sandwich compression (distinct layout, Rcpp) | 0.56 |

---

## 8. Known gaps, deferred items, and errata

- The CSS delta-method SE ignores the uncertainty in $\hat e$ and $\bar x_C$. A **linearized closed form from compressed rows** was verified in research: take the complex-step gradient of the effect with respect to the stratum sums $(\sum\xi, \sum\xi u)$ at $\xi=1$, then $\text{Var}=\sum_h c_h^\top(\sum v v^\top)c_h$. It matched the Bayesian bootstrap SD (0.0585 vs 0.0583). It is not in either post. **It is a good candidate for the library's default SE.**
- Second-order approximation: its scope should be stated in the API (effects at $\bar x_C$ only).
- YOCO paper (arXiv 2102.11297) errata found earlier: the logistic vcov needs the inverse and the sigmoid (not logit); frequency weights need $\sum w-p$ degrees of freedom.
- Not researched yet: small-sample corrections (CR1/CR2) in compressed form; missing outcomes; staggered designs with time-varying covariates; weak first stage diagnostics (first-stage F from compressed data is possible but unverified); third-moment correction for (b).

---

## 9. Working agreement with the user (put the essentials in the new repo's `CLAUDE.md`)

The user is Jeffrey Wong (`jeff.ct.wong@gmail.com`), an experimentation and causal inference expert who expects graduate-level rigor.
- **Explain new derivations in chat first. Change code that encodes math only after an explicit yes.**
- **Verify before asserting.** Never tune a simulation to tell a story. Never cite a result you haven't checked against the source. Phrase findings as "targets X" with the derivation rather than "X is wrong".
- **Plan before building.** For large work: propose the architecture, API, naming table and test plan, and get a yes. Use `AskUserQuestion` for decisions that are genuinely the user's.
- **Naming discipline.** One meaning per symbol or name. Code names mirror the math (`beta`, `x_C`, `xi`, `G_h`, `first_moments`). Keep the vocabulary: profile row → account summary → compressed row → compressed dataset.
- **Performance is a feature.** Profile first. Cap parallelism (the user watches CPU and memory; the research used `min(4, cores)`).
- **R style, if R:** native pipe with one `|>` per line, purrr instead of loops, `<-`.
- **Commits:** only when asked. Branch first if on the default branch.

---

## 10. Open decisions to confirm with the user before coding

1. **Language and packaging.** Options include:
   - an R package (closest to the reference; Rcpp kernels port directly);
   - Python with a C++ or Rust core (numpy, pybind11 or PyO3);
   - a C++ core with both bindings;
   - SQL/Spark UDAFs for the compression step only.
2. **Execution model.** In-memory, streaming, or distributed (Spark/Beam/BigQuery). This decides how "rows grouped by account" is guaranteed: a sort, or a shuffle by account key.
3. **RNG.** Which counter-based generator, and how (seed, account id, draw) maps to a stream.
4. **Which uncertainty is the default.**
   - Sandwich: CSS-style, or linearized including $\hat e$ and $\bar x_C$.
   - Exact posterior.
   - Compressed posterior.
5. **API surface.** Suggested layering:
   - a `Design` spec: unit/cluster column, strata columns, instrument, treatment, covariates with $x_{ref}$, outcome columns, effect functions;
   - a mergeable `Compressor` state with `update(rows)`, `merge(other)`, `serialize()`;
   - `fit(compressed)` → coefficients and covariance;
   - `effects(fit, delta / effect_fn)`;
   - `posterior(compressed or rows, B, seed, engine = {"compressed", "exact"})`.
6. **Storage format** for compressed rows: distinct layout, versioned schema, with $x_{ref}$ and the layout recorded in metadata.
7. **Validation harness.** Port the simulator and the estimatr oracle, or call R from the test suite.

---

## 11. Suggested milestone plan (propose to the user; don't start without a yes)

1. **Reference port.**
   - Port `simulate_accounts`, `prepare`, `fit_rows`, `complier_mean`, the effects (delta vectors and gradients), and `draw_rows` to the target language.
   - Pass invariants 1 (rows vs textbook), 3 and 7.
2. **Compression.**
   - Implement the one-pass buffer-and-flush compressor with the distinct layout, the mergeable state, `fit_sums`, and the sandwich from second moments.
   - Pass invariants 1, 2 and 8.
3. **Bootstrap engines.**
   - Implement the counter-based weights, the exact one-pass engine with the contiguous-over-$b$ layout, and the compressed Gaussian engine.
   - Pass invariants 4, 5 and 6, and reproduce the calibration table within Monte Carlo error.
4. **Scale.**
   - Distributed compression, batched per-draw solves, and benchmarks against the Section 7 performance table.
5. **Extensions.** Prognostic scores, the second-order approximation, the linearized SE including $\hat e$ and $\bar x_C$, and small-sample corrections.
