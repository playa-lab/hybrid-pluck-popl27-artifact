# Comparison with the paper: `reduced_20261010_004546`

Tier `reduced` on aarch64 with 8 CPUs; timeouts: pluck 600 s, psi 300 s, hybit 300 s, slicstan 300 s, siren 300 s, turing 300 s.

## Summary

- Claims: 10 of 10 checked claims agree.
- Tables 2 and 3: 219 cells ran; 201 have the paper's status, 11 timed out here but finished in the paper, 0 were limited by this machine's memory, 6 are on the other side of the 5% error threshold, and 1 differ otherwise. 4 cells were not run.
- Tables 4 and 5: 0 counts differ from the paper, and 19 SPN sizes that vary between runs; 6 programs have no counts in this run.

## The paper's claims

| Claim | Paper reference | Paper | This run | Agrees |
|---|---|---|---|---|
| Hybrid Pluck finishes on problems where the exact baseline (PSI) times out | Abstract (14 problems) | 13 models in the rows run here (14 in all rows) | 15 models (adds Polynomial Regression, Secret Ballot Vote) | yes |
| Pluck is the fastest deterministic method on almost all standard benchmarks (rows where it is not) | §6.2, Table 2 | diamond_network | none | yes |
| Pluck is faster than PSI and HyBit on every Table 3 row (rows where it is not) | §6.2, Table 3 | f81_rate | f81_rate | yes |
| Siren is consistently the best-performing approximate method (rows where it is fastest) | §6.3, Table 3 | 25 of 28 | 22 of 26 | yes |
| Turing fails to converge on Ecological Inference (r/e or t/o) | §6.3, Fig. 9b | 2 of 2 rows | 2 of 2 rows | yes |
| Tables 4 and 5's BDD and SPN sizes match exactly (except the SPN sizes in lists/spn_varies.txt) | Tables 4, 5 | 505 cells | 505 match | yes |
| Figure 1's programs have the paper's lines of code | Fig. 1 | 5 programs | 5 match | yes |
| PSI times out on all five Figure 1 models | Fig. 1 caption | t/o at 600 s | 5 of 5 t/o at 600 s | yes |
| The overview example's numbers | §2.4 | Z 4.9e-03; P(s2) 0.57; iota 0.143 -> 0.178; rho 0.333 -> 0.329 | Z 4.9e-03; P(s2) 0.57; iota 0.143 -> 0.178; rho 0.333 -> 0.329 | yes |
| This run's Pluck answers equal the fixed exact answers (largest relative difference) | reference/exact_answers.csv | 0 | 7.1e-15 over 91 programs | yes |

## Differences from the paper

### Timed out here, finished in the paper

Not a disagreement: this machine (or this tier's timeout) is slower than the paper's. The note gives the slowdown this implies.

| Table | Program | System | Paper (s) | This run | Note |
|---|---|---|---|---|---|
| Table 2 | normal_mixture | hybit | 43.5 | t/o | at least 6.9× slower than the paper |
| Table 2 | addFun_max | hybit | 514 | t/o | the paper's time is above this run's 300 s timeout |
| Table 2 | tug_of_war | hybit | 583 | t/o | the paper's time is above this run's 300 s timeout |
| Table 3 | polyreg | psi | 111 | t/o | at least 2.7× slower than the paper |
| Table 3 | changepoint_coal_n020 | hybit | r/e | t/o | the paper's run finished, with error above 5% |
| Table 3 | hmm_first_order_n08 | hybit | r/e | t/o | the paper's run finished, with error above 5% |
| Table 3 | hmm_second_order_n04 | hybit | 77 | t/o | at least 3.9× slower than the paper |
| Table 3 | king_ei_d60 | hybit | 137 | t/o | at least 2.2× slower than the paper |
| Table 3 | outliers_n08 | psi | 141 | t/o | at least 2.1× slower than the paper |
| Table 3 | outliers_n08 | hybit | r/e | t/o | the paper's run finished, with error above 5% |
| Table 3 | secret_ballot_n04 | psi | 215 | t/o | at least 1.4× slower than the paper |

### Limited by this machine

Pluck programs in `lists/large.txt`, which need more than 10 GB of memory; see README.md for each one's memory and time.

None.

### Approximate inference on the other side of the 5% error threshold

Sampling is random, and smaller tiers sweep fewer budgets, so a cell near the threshold can land on either side.

| Table | Program | System | Paper (s) | This run | Note |
|---|---|---|---|---|---|
| Table 3 | sir_model_n12 | siren | 6.4 | r/e |  |
| Table 3 | sir_model_n72 | siren | 210 | r/e |  |
| Table 3 | changepoint_coal_n050 | turing | 575 | r/e |  |
| Table 3 | dawid_skene_n16 | turing | r/e | 51.5 |  |
| Table 3 | hmm_second_order_n08 | turing | 30.2 | r/e |  |
| Table 3 | secret_ballot_n128 | siren | 328 | r/e |  |

### Other differences

| Table | Program | System | Paper (s) | This run | Note |
|---|---|---|---|---|---|
| Table 3 | pcfg_ambiguous | turing | t/o | r/e |  |

### Tables 4 and 5: counts that differ

None.

Programs with no counts in this run: changepoint_coal_n050, king_ei_d15, king_ei_d20, secret_ballot_n128, sir_model_n72, soccer_n08.

### Tables 4 and 5: SPN sizes that vary between runs

The programs in `lists/spn_varies.txt` build SPNs of different sizes on different runs or machines; their BDD counts and answers do not change. Not a disagreement.

| Table | Program | Count | Paper | This run | Difference |
|---|---|---|---|---|---|
| Table 4 | secret_ballot_n08 | spn_interned | 2,006 | 2,014 | +0.40% |
| Table 4 | secret_ballot_n08 | spn_tree | 59,583 | 62,115 | +4.25% |
| Table 4 | secret_ballot_n08 | spn_compression | 30 | 31 | +3.33% |
| Table 4 | secret_ballot_n16 | spn_interned | 19,564 | 19,554 | -0.05% |
| Table 4 | secret_ballot_n16 | spn_tree | 636,666,000 | 623,933,000 | -2.00% |
| Table 4 | secret_ballot_n16 | spn_compression | 32,543 | 31,908 | -1.95% |
| Table 4 | secret_ballot_n32 | spn_interned | 115,052 | 114,982 | -0.06% |
| Table 4 | secret_ballot_n64 | spn_interned | 548,176 | 548,322 | +0.03% |
| Table 5 | dawid_skene_n08 | spn_interned | 157 | 159 | +1.27% |
| Table 5 | dawid_skene_n18 | spn_interned | 9,469 | 10,528 | +11.18% |
| Table 5 | dawid_skene_n18 | spn_tree | 253,126 | 287,845 | +13.72% |
| Table 5 | secret_ballot_n08 | spn_interned | 2,006 | 2,014 | +0.40% |
| Table 5 | secret_ballot_n08 | spn_tree | 59,583 | 62,115 | +4.25% |
| Table 5 | secret_ballot_n08 | spn_compression | 30 | 31 | +3.33% |
| Table 5 | secret_ballot_n16 | spn_interned | 19,564 | 19,554 | -0.05% |
| Table 5 | secret_ballot_n16 | spn_tree | 636,666,000 | 623,933,000 | -2.00% |
| Table 5 | secret_ballot_n16 | spn_compression | 32,543 | 31,908 | -1.95% |
| Table 5 | secret_ballot_n32 | spn_interned | 115,052 | 114,982 | -0.06% |
| Table 5 | secret_ballot_n64 | spn_interned | 548,176 | 548,322 | +0.03% |

## Timing: this run / paper

Geometric mean, smallest and largest ratio of this run's time to the paper's, over the cells where both have a time. Approximate systems' times are those of the budget each run selected.

| Table | Metric | Cells | Geometric mean | Min | Max |
|---|---|---|---|---|---|
| Fig. 1 | time_s | 5 | 1.08× | 0.79× | 1.44× |
| Table 2 | hybit | 13 | 2.39× | 1.08× | 9.14× |
| Table 2 | pluck | 12 | 1.11× | 0.08× | 2.40× |
| Table 2 | psi | 14 | 1.16× | 0.68× | 1.76× |
| Table 3 | hybit | 2 | 2.05× | 1.92× | 2.18× |
| Table 3 | pluck | 29 | 1.80× | 0.81× | 12.76× |
| Table 3 | psi | 6 | 2.00× | 1.16× | 4.38× |
| Table 3 | siren | 22 | 1.62× | 0.26× | 16.72× |
| Table 3 | slicstan | 8 | 1.05× | 1.00× | 1.11× |
| Table 3 | turing | 17 | 1.35× | 0.36× | 5.39× |
| Table 4 | t_posterior | 16 | 1.32× | 0.50× | 3.14× |
| Table 4 | t_symbolic | 16 | 1.67× | 1.06× | 3.17× |
| Table 4 | t_wmc | 16 | 1.92× | 0.85× | 7.56× |
| Table 5 | t_posterior | 71 | 1.45× | 0.50× | 4.40× |
| Table 5 | t_symbolic | 90 | 1.74× | 0.11× | 14.96× |
| Table 5 | t_wmc | 82 | 1.54× | 0.39× | 7.56× |

## Figure 1

Compare the panels in `fig1/` with the paper's in `experiments/reference/fig1/`. The fonts differ (the image has no Linux Libertine).

| Program | LoC paper | LoC here | Time paper (s) | Time here (s) | PSI here |
|---|---|---|---|---|---|
| fig_changepoint_listing | 40 | 40 | 0.9 | 1.3 | t/o |
| fig_dawid_skene_listing | 39 | 39 | 0.016 | 0.018 | t/o |
| fig_mixture_listing | 32 | 32 | 1.4 | 1.1 | t/o |
| fig_polyreg_listing | 35 | 35 | 1.3 | 1.5 | t/o |
| fig_sir_listing | 52 | 52 | 0.9 | 0.9 | t/o |

## Every cell of Tables 2 and 3

| Table | Program | System | Paper | This run | This run / paper |
|---|---|---|---|---|---|
| Table 2 | burglary_alarm | pluck | 1.26e-04 | 1.18e-04 | 0.94× |
| Table 2 | burglary_alarm | psi | 0.0883 | 0.0796 | 0.90× |
| Table 2 | burglary_alarm | hybit | 2.56e-03 | 0.0234 | 9.14× |
| Table 2 | burglary_alarm | hybit_err | blank | blank |  |
| Table 2 | diamond_network | pluck | 0.0121 | 0.0231 | 1.91× |
| Table 2 | diamond_network | psi | 1.9 | 2.81 | 1.48× |
| Table 2 | diamond_network | hybit | 9.08e-03 | 0.0291 | 3.20× |
| Table 2 | diamond_network | hybit_err | blank | blank |  |
| Table 2 | typos | pluck | 0.0166 | 0.0298 | 1.80× |
| Table 2 | addFun_sum | pluck | 6.54e-05 | 6.58e-05 | 1.01× |
| Table 2 | addFun_sum | psi | 0.017 | 0.0115 | 0.68× |
| Table 2 | addFun_sum | hybit | 0.311 | 0.533 | 1.71× |
| Table 2 | addFun_sum | hybit_err | 3.90e-03 | 3.90e-03 | 1.00× |
| Table 2 | conjugate_gaussians | pluck | 1.21e-04 | 1.97e-04 | 1.63× |
| Table 2 | conjugate_gaussians | psi | 0.018 | 0.0255 | 1.42× |
| Table 2 | conjugate_gaussians | hybit | 1.07 | 3.02 | 2.82× |
| Table 2 | conjugate_gaussians | hybit_err | 7.00e-04 | 7.00e-04 | 1.00× |
| Table 2 | linear_regression | pluck | 4.15e-04 | 6.81e-04 | 1.64× |
| Table 2 | linear_regression | psi | 0.632 | 0.879 | 1.39× |
| Table 2 | linear_regression | hybit | 3.19 | 6.27 | 1.97× |
| Table 2 | linear_regression | hybit_err | 3.40e-03 | 3.40e-03 | 1.00× |
| Table 2 | clickGraph | pluck | 5.64e-04 | 7.57e-04 | 1.34× |
| Table 2 | clickGraph | psi | 3.22 | 3.56 | 1.11× |
| Table 2 | clickGraph | hybit | 0.48 | 2.14 | 4.46× |
| Table 2 | clickGraph | hybit_err | 0.0228 | 0.0228 | 1.00× |
| Table 2 | clinicalTrial1 | pluck | 0.0359 | 0.0464 | 1.29× |
| Table 2 | clinicalTrial1 | psi | t/o | t/o |  |
| Table 2 | clinicalTrial1 | hybit | 3.83 | 5.95 | 1.55× |
| Table 2 | clinicalTrial1 | hybit_err | 1.40e-03 | 1.40e-03 | 1.00× |
| Table 2 | clinicalTrial2 | pluck | 2.80e-04 | 3.01e-04 | 1.08× |
| Table 2 | clinicalTrial2 | psi | 0.438 | 0.771 | 1.76× |
| Table 2 | clinicalTrial2 | hybit | 0.241 | 0.377 | 1.56× |
| Table 2 | clinicalTrial2 | hybit_err | 6.00e-04 | 6.00e-04 | 1.00× |
| Table 2 | coinBias | pluck | 1.72e-04 | 1.93e-04 | 1.12× |
| Table 2 | coinBias | psi | 0.201 | 0.322 | 1.60× |
| Table 2 | coinBias | hybit | 0.0449 | 0.14 | 3.12× |
| Table 2 | coinBias | hybit_err | 2.00e-03 | 2.00e-03 | 1.00× |
| Table 2 | GPA | pluck | 1.64e-03 | 1.36e-04 | 0.08× |
| Table 2 | GPA | psi | 0.915 | 0.946 | 1.03× |
| Table 2 | GPA | hybit | 0.171 | 0.618 | 3.61× |
| Table 2 | GPA | hybit_err | 0 | 0 |  |
| Table 2 | normal_mixture | pluck | 2.89 | 6.93 | 2.40× |
| Table 2 | normal_mixture | psi | t/o | t/o |  |
| Table 2 | normal_mixture | hybit | 43.5 | t/o |  |
| Table 2 | addFun_max | psi | 0.0209 | 0.0163 | 0.78× |
| Table 2 | addFun_max | hybit | 514 | t/o |  |
| Table 2 | spacex | psi | 0.102 | 0.149 | 1.46× |
| Table 2 | spacex | hybit | 21.1 | 28.7 | 1.36× |
| Table 2 | spacex | hybit_err | 2.00e-04 | 2.00e-04 | 1.00× |
| Table 2 | trueskill | psi | 0.129 | 0.121 | 0.94× |
| Table 2 | trueskill | hybit | 116 | 197 | 1.70× |
| Table 2 | trueskill | hybit_err | 6.20e-03 | 6.20e-03 | 1.00× |
| Table 2 | tug_of_war | psi | 11.5 | 11.7 | 1.02× |
| Table 2 | tug_of_war | hybit | 583 | t/o |  |
| Table 2 | weekend | psi | 0.0745 | 0.0938 | 1.26× |
| Table 2 | weekend | hybit | 0.149 | 0.161 | 1.08× |
| Table 2 | weekend | hybit_err | 1.00e-04 | 1.00e-04 | 1.00× |
| Table 3 | spike_and_slab | pluck | 6.73e-03 | 0.0134 | 1.99× |
| Table 3 | spike_and_slab | psi | 21.1 | 44.5 | 2.11× |
| Table 3 | spike_and_slab | hybit | t/o | t/o |  |
| Table 3 | spike_and_slab | siren | t/o | t/o |  |
| Table 3 | spike_and_slab | turing | r/e | r/e |  |
| Table 3 | polyreg | pluck | 0.0105 | 0.0321 | 3.06× |
| Table 3 | polyreg | psi | 111 | t/o |  |
| Table 3 | polyreg | hybit | t/o | t/o |  |
| Table 3 | polyreg | slicstan | r/e | r/e |  |
| Table 3 | polyreg | siren | t/o | t/o |  |
| Table 3 | polyreg | turing | r/e | r/e |  |
| Table 3 | run_walk_hrm | pluck | 6.15e-03 | 0.0107 | 1.74× |
| Table 3 | run_walk_hrm | psi | t/o | t/o |  |
| Table 3 | run_walk_hrm | hybit | t/o | t/o |  |
| Table 3 | run_walk_hrm | slicstan | r/e | r/e |  |
| Table 3 | run_walk_hrm | siren | 0.0699 | 0.31 | 4.43× |
| Table 3 | run_walk_hrm | turing | 1.06 | 1.12 | 1.06× |
| Table 3 | sir_model_n12 | pluck | 0.0105 | 8.54e-03 | 0.81× |
| Table 3 | sir_model_n12 | psi | t/o | t/o |  |
| Table 3 | sir_model_n12 | hybit | r/e | r/e |  |
| Table 3 | sir_model_n12 | slicstan | 10.4 | 10.6 | 1.02× |
| Table 3 | sir_model_n12 | siren | 6.4 | r/e |  |
| Table 3 | sir_model_n12 | turing | 14.3 | 12 | 0.84× |
| Table 3 | sir_model_n72 | pluck | 121 | n/r |  |
| Table 3 | sir_model_n72 | psi | t/o | t/o |  |
| Table 3 | sir_model_n72 | hybit | t/o | t/o |  |
| Table 3 | sir_model_n72 | slicstan | t/o | t/o |  |
| Table 3 | sir_model_n72 | siren | 210 | r/e |  |
| Table 3 | sir_model_n72 | turing | r/e | r/e |  |
| Table 3 | f81_rate | pluck | 0.092 | 0.144 | 1.57× |
| Table 3 | f81_rate | psi | t/o | t/o |  |
| Table 3 | f81_rate | hybit | 0.0619 | 0.135 | 2.18× |
| Table 3 | f81_rate | turing | r/e | r/e |  |
| Table 3 | f81_joint | pluck | 0.907 | 1.25 | 1.38× |
| Table 3 | f81_joint | psi | t/o | t/o |  |
| Table 3 | f81_joint | turing | r/e | r/e |  |
| Table 3 | pcfg_ambiguous | pluck | 0.276 | 0.427 | 1.55× |
| Table 3 | pcfg_ambiguous | turing | t/o | r/e |  |
| Table 3 | pcfg_n020 | pluck | 0.25 | 0.368 | 1.47× |
| Table 3 | pcfg_n020 | turing | 37.4 | 13.3 | 0.36× |
| Table 3 | changepoint_coal_k1 | pluck | 0.0144 | 0.0166 | 1.15× |
| Table 3 | changepoint_coal_k1 | psi | 17.6 | 20.4 | 1.16× |
| Table 3 | changepoint_coal_k1 | hybit | 2.59 | 4.97 | 1.92× |
| Table 3 | changepoint_coal_k1 | slicstan | 2.88 | 3.16 | 1.10× |
| Table 3 | changepoint_coal_k1 | siren | 0.0733 | 0.292 | 3.98× |
| Table 3 | changepoint_coal_k1 | turing | 2.32 | 12.5 | 5.39× |
| Table 3 | changepoint_coal_n020 | pluck | 0.684 | 0.742 | 1.08× |
| Table 3 | changepoint_coal_n020 | psi | t/o | t/o |  |
| Table 3 | changepoint_coal_n020 | hybit | r/e | t/o |  |
| Table 3 | changepoint_coal_n020 | siren | 18.6 | 10 | 0.54× |
| Table 3 | changepoint_coal_n020 | turing | r/e | r/e |  |
| Table 3 | changepoint_coal_n050 | pluck | 178 | n/r |  |
| Table 3 | changepoint_coal_n050 | psi | t/o | t/o |  |
| Table 3 | changepoint_coal_n050 | hybit | t/o | t/o |  |
| Table 3 | changepoint_coal_n050 | siren | 8.26 | 8.64 | 1.05× |
| Table 3 | changepoint_coal_n050 | turing | 575 | r/e |  |
| Table 3 | dawid_skene_n08 | pluck | 3.28e-03 | 3.07e-03 | 0.94× |
| Table 3 | dawid_skene_n08 | psi | t/o | t/o |  |
| Table 3 | dawid_skene_n08 | hybit | t/o | t/o |  |
| Table 3 | dawid_skene_n08 | slicstan | 27.8 | 27.7 | 1.00× |
| Table 3 | dawid_skene_n08 | siren | 0.217 | 0.2 | 0.92× |
| Table 3 | dawid_skene_n08 | turing | 15.1 | 40.3 | 2.67× |
| Table 3 | dawid_skene_n16 | pluck | 0.0612 | 0.141 | 2.30× |
| Table 3 | dawid_skene_n16 | psi | t/o | t/o |  |
| Table 3 | dawid_skene_n16 | hybit | t/o | t/o |  |
| Table 3 | dawid_skene_n16 | slicstan | t/o | t/o |  |
| Table 3 | dawid_skene_n16 | siren | 6.47 | 1.65 | 0.26× |
| Table 3 | dawid_skene_n16 | turing | r/e | 51.5 |  |
| Table 3 | hmm_first_order_n02 | pluck | 5.17e-04 | 4.21e-04 | 0.81× |
| Table 3 | hmm_first_order_n02 | psi | 0.644 | 1.11 | 1.72× |
| Table 3 | hmm_first_order_n02 | hybit | r/e | r/e |  |
| Table 3 | hmm_first_order_n02 | slicstan | 0.346 | 0.357 | 1.03× |
| Table 3 | hmm_first_order_n02 | siren | 0.0257 | 0.0508 | 1.98× |
| Table 3 | hmm_first_order_n02 | turing | 1.39 | 4.2 | 3.02× |
| Table 3 | hmm_first_order_n08 | pluck | 0.0418 | 0.0708 | 1.69× |
| Table 3 | hmm_first_order_n08 | psi | t/o | t/o |  |
| Table 3 | hmm_first_order_n08 | hybit | r/e | t/o |  |
| Table 3 | hmm_first_order_n08 | slicstan | 3.24 | 3.59 | 1.11× |
| Table 3 | hmm_first_order_n08 | siren | 0.885 | 0.782 | 0.88× |
| Table 3 | hmm_first_order_n08 | turing | 31.2 | 30.6 | 0.98× |
| Table 3 | hmm_first_order_n14 | pluck | 43.8 | 122 | 2.79× |
| Table 3 | hmm_first_order_n14 | psi | t/o | t/o |  |
| Table 3 | hmm_first_order_n14 | hybit | t/o | t/o |  |
| Table 3 | hmm_first_order_n14 | slicstan | 17.8 | 18.6 | 1.04× |
| Table 3 | hmm_first_order_n14 | siren | 0.81 | 0.357 | 0.44× |
| Table 3 | hmm_first_order_n14 | turing | 10 | 12.1 | 1.21× |
| Table 3 | hmm_second_order_n04 | pluck | 1.72e-03 | 2.56e-03 | 1.49× |
| Table 3 | hmm_second_order_n04 | psi | 35.6 | 156 | 4.38× |
| Table 3 | hmm_second_order_n04 | hybit | 77 | t/o |  |
| Table 3 | hmm_second_order_n04 | siren | 0.0675 | 0.125 | 1.85× |
| Table 3 | hmm_second_order_n04 | turing | 1.03 | 1.56 | 1.51× |
| Table 3 | hmm_second_order_n08 | pluck | 0.0473 | 0.0678 | 1.43× |
| Table 3 | hmm_second_order_n08 | psi | t/o | t/o |  |
| Table 3 | hmm_second_order_n08 | hybit | t/o | t/o |  |
| Table 3 | hmm_second_order_n08 | siren | 0.478 | 3.85 | 8.05× |
| Table 3 | hmm_second_order_n08 | turing | 30.2 | r/e |  |
| Table 3 | hmm_second_order_n14 | pluck | 47.1 | 147 | 3.12× |
| Table 3 | hmm_second_order_n14 | psi | t/o | t/o |  |
| Table 3 | hmm_second_order_n14 | hybit | t/o | t/o |  |
| Table 3 | hmm_second_order_n14 | siren | 0.482 | 0.416 | 0.86× |
| Table 3 | hmm_second_order_n14 | turing | 10.3 | 18.2 | 1.77× |
| Table 3 | hmm_factorial_n02 | pluck | 1.71e-03 | 1.46e-03 | 0.85× |
| Table 3 | hmm_factorial_n02 | psi | 2.23 | 4.85 | 2.17× |
| Table 3 | hmm_factorial_n02 | hybit | t/o | t/o |  |
| Table 3 | hmm_factorial_n02 | siren | 0.043 | 0.078 | 1.81× |
| Table 3 | hmm_factorial_n02 | turing | 1.19 | 1.84 | 1.55× |
| Table 3 | hmm_factorial_n04 | pluck | 0.0142 | 0.0212 | 1.49× |
| Table 3 | hmm_factorial_n04 | psi | t/o | t/o |  |
| Table 3 | hmm_factorial_n04 | hybit | t/o | t/o |  |
| Table 3 | hmm_factorial_n04 | siren | 0.142 | 0.153 | 1.08× |
| Table 3 | hmm_factorial_n04 | turing | 3.52 | 8.2 | 2.33× |
| Table 3 | hmm_factorial_n06 | pluck | 0.37 | 0.466 | 1.26× |
| Table 3 | hmm_factorial_n06 | psi | t/o | t/o |  |
| Table 3 | hmm_factorial_n06 | hybit | t/o | t/o |  |
| Table 3 | hmm_factorial_n06 | siren | 0.102 | 0.231 | 2.26× |
| Table 3 | hmm_factorial_n06 | turing | 4.35 | 4.32 | 0.99× |
| Table 3 | king_ei_d60 | pluck | 0.329 | 0.516 | 1.57× |
| Table 3 | king_ei_d60 | psi | t/o | t/o |  |
| Table 3 | king_ei_d60 | hybit | 137 | t/o |  |
| Table 3 | king_ei_d60 | siren | 3.73 | 1.61 | 0.43× |
| Table 3 | king_ei_d60 | turing | r/e | r/e |  |
| Table 3 | king_ei_d20 | pluck | 22.6 | n/r |  |
| Table 3 | king_ei_d20 | psi | t/o | t/o |  |
| Table 3 | king_ei_d20 | hybit | t/o | t/o |  |
| Table 3 | king_ei_d20 | siren | 4.34 | 1.81 | 0.42× |
| Table 3 | king_ei_d20 | turing | r/e | r/e |  |
| Table 3 | outliers_n08 | pluck | 6.56e-03 | 0.0151 | 2.30× |
| Table 3 | outliers_n08 | psi | 141 | t/o |  |
| Table 3 | outliers_n08 | hybit | r/e | t/o |  |
| Table 3 | outliers_n08 | slicstan | 3.25 | 3.32 | 1.02× |
| Table 3 | outliers_n08 | siren | 0.241 | 0.964 | 4.00× |
| Table 3 | outliers_n08 | turing | 13.4 | 7.16 | 0.53× |
| Table 3 | outliers_n18 | pluck | 18.4 | 87.9 | 4.78× |
| Table 3 | outliers_n18 | psi | t/o | t/o |  |
| Table 3 | outliers_n18 | hybit | t/o | t/o |  |
| Table 3 | outliers_n18 | slicstan | 39.2 | 41 | 1.05× |
| Table 3 | outliers_n18 | siren | 1.09 | 4.5 | 4.13× |
| Table 3 | outliers_n18 | turing | 25.4 | 59.3 | 2.33× |
| Table 3 | soft_kmeans_n04 | pluck | 3.30e-03 | 5.56e-03 | 1.68× |
| Table 3 | soft_kmeans_n04 | psi | 18 | 29.1 | 1.62× |
| Table 3 | soft_kmeans_n04 | hybit | t/o | t/o |  |
| Table 3 | soft_kmeans_n04 | slicstan | r/e | r/e |  |
| Table 3 | soft_kmeans_n04 | siren | 0.0473 | 0.791 | 16.72× |
| Table 3 | soft_kmeans_n04 | turing | r/e | r/e |  |
| Table 3 | soft_kmeans_n12 | pluck | 21.1 | 75 | 3.55× |
| Table 3 | soft_kmeans_n12 | psi | t/o | t/o |  |
| Table 3 | soft_kmeans_n12 | hybit | t/o | t/o |  |
| Table 3 | soft_kmeans_n12 | slicstan | r/e | r/e |  |
| Table 3 | soft_kmeans_n12 | siren | 0.14 | 0.657 | 4.69× |
| Table 3 | soft_kmeans_n12 | turing | r/e | r/e |  |
| Table 3 | secret_ballot_n04 | pluck | 1.33e-03 | 2.04e-03 | 1.53× |
| Table 3 | secret_ballot_n04 | psi | 215 | t/o |  |
| Table 3 | secret_ballot_n04 | hybit | t/o | t/o |  |
| Table 3 | secret_ballot_n04 | siren | 0.241 | 0.43 | 1.78× |
| Table 3 | secret_ballot_n04 | turing | 1.72 | 0.625 | 0.36× |
| Table 3 | secret_ballot_n128 | pluck | 71.2 | n/r |  |
| Table 3 | secret_ballot_n128 | psi | t/o | t/o |  |
| Table 3 | secret_ballot_n128 | hybit | t/o | t/o |  |
| Table 3 | secret_ballot_n128 | siren | 328 | r/e |  |
| Table 3 | secret_ballot_n128 | turing | r/e | r/e |  |
| Table 3 | soccer_n03 | pluck | 4.95e-03 | 0.0124 | 2.51× |
| Table 3 | soccer_n03 | psi | t/o | t/o |  |
| Table 3 | soccer_n03 | turing | 11.8 | 26.8 | 2.27× |
| Table 3 | soccer_n07 | pluck | 7.1 | 90.6 | 12.76× |
| Table 3 | soccer_n07 | psi | t/o | t/o |  |
| Table 3 | soccer_n07 | turing | r/e | r/e |  |
