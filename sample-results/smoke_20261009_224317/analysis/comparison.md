# Comparison with the paper: `smoke_20261009_224317`

Tier `smoke` on aarch64 with 8 CPUs; timeouts: pluck 120 s, psi 120 s, hybit 120 s, slicstan 120 s, siren 120 s, turing 120 s.

## Summary

- Claims: 8 of 8 checked claims agree.
- Tables 2 and 3: 10 cells ran; 10 have the paper's status, 0 timed out here but finished in the paper, 0 were limited by this machine's memory, 0 are on the other side of the 5% error threshold, and 0 differ otherwise. 203 cells were not run.
- Tables 4 and 5: 0 counts differ from the paper, and 0 SPN sizes that vary between runs; 95 programs have no counts in this run.

## The paper's claims

| Claim | Paper reference | Paper | This run | Agrees |
|---|---|---|---|---|
| Pluck is the fastest deterministic method on almost all standard benchmarks (rows where it is not) | §6.2, Table 2 | none | none | yes |
| Pluck is faster than PSI and HyBit on every Table 3 row (rows where it is not) | §6.2, Table 3 | none | none | yes |
| Siren is consistently the best-performing approximate method (rows where it is fastest) | §6.3, Table 3 | 1 of 1 | 1 of 1 | yes |
| Tables 4 and 5's BDD and SPN sizes match exactly (except the SPN sizes in lists/spn_varies.txt) | Tables 4, 5 | 10 cells | 10 match | yes |
| Figure 1's programs have the paper's lines of code | Fig. 1 | 5 programs | 5 match | yes |
| PSI times out on all five Figure 1 models | Fig. 1 caption | t/o at 600 s | 5 of 5 t/o at 60 s | yes |
| The overview example's numbers | §2.4 | Z 4.9e-03; P(s2) 0.57; iota 0.143 -> 0.178; rho 0.333 -> 0.329 | Z 4.9e-03; P(s2) 0.57; iota 0.143 -> 0.178; rho 0.333 -> 0.329 | yes |
| This run's Pluck answers equal the fixed exact answers (largest relative difference) | reference/exact_answers.csv | 0 | 0.0e+00 over 2 programs | yes |

## Differences from the paper

### Timed out here, finished in the paper

Not a disagreement: this machine (or this tier's timeout) is slower than the paper's. The note gives the slowdown this implies.

None.

### Limited by this machine

Pluck programs in `lists/large.txt`, which need more than 10 GB of memory; see README.md for each one's memory and time.

None.

### Approximate inference on the other side of the 5% error threshold

Sampling is random, and smaller tiers sweep fewer budgets, so a cell near the threshold can land on either side.

None.

### Other differences

None.

### Tables 4 and 5: counts that differ

None.

Programs with no counts in this run: GPA, addFun_sum, burglary_alarm, changepoint_coal_n010, changepoint_coal_n015, changepoint_coal_n020, changepoint_coal_n025, changepoint_coal_n030, changepoint_coal_n035, changepoint_coal_n040, changepoint_coal_n050, clickGraph, clinicalTrial1, clinicalTrial2, conjugate_gaussians, dawid_skene_n08, dawid_skene_n12, dawid_skene_n16, dawid_skene_n18, dawid_skene_n20, dawid_skene_n22, diamond_network, f81_joint, f81_rate, hmm_factorial_n02, hmm_factorial_n03, hmm_factorial_n04, hmm_factorial_n05, hmm_factorial_n06, hmm_factorial_n07, hmm_first_order_n02, hmm_first_order_n04, hmm_first_order_n06, hmm_first_order_n08, hmm_first_order_n10, hmm_first_order_n12, hmm_first_order_n14, hmm_second_order_n02, hmm_second_order_n04, hmm_second_order_n06, hmm_second_order_n08, hmm_second_order_n10, hmm_second_order_n12, hmm_second_order_n14, king_ei_d15, king_ei_d20, king_ei_d25, king_ei_d30, king_ei_d40, king_ei_d60, king_ei_d80, linear_regression, normal_mixture, outliers_n08, outliers_n10, outliers_n12, outliers_n14, outliers_n16, outliers_n18, pcfg_ambiguous, pcfg_n010, pcfg_n020, pcfg_n040, pcfg_n060, pcfg_n080, pcfg_n100, polyreg, run_walk_hrm, secret_ballot_n04, secret_ballot_n08, secret_ballot_n128, secret_ballot_n16, secret_ballot_n32, secret_ballot_n64, sir_model_n12, sir_model_n24, sir_model_n36, sir_model_n48, sir_model_n60, sir_model_n72, soccer_n03, soccer_n04, soccer_n05, soccer_n06, soccer_n07, soccer_n08, soft_kmeans_n02, soft_kmeans_n04, soft_kmeans_n06, soft_kmeans_n08, soft_kmeans_n10, soft_kmeans_n11, soft_kmeans_n12, spike_and_slab, typos.

### Tables 4 and 5: SPN sizes that vary between runs

The programs in `lists/spn_varies.txt` build SPNs of different sizes on different runs or machines; their BDD counts and answers do not change. Not a disagreement.

None.

## Timing: this run / paper

Geometric mean, smallest and largest ratio of this run's time to the paper's, over the cells where both have a time. Approximate systems' times are those of the budget each run selected.

| Table | Metric | Cells | Geometric mean | Min | Max |
|---|---|---|---|---|---|
| Fig. 1 | time_s | 5 | 1.07× | 0.79× | 1.56× |
| Table 2 | hybit | 1 | 0.74× | 0.74× | 0.74× |
| Table 2 | pluck | 1 | 1.54× | 1.54× | 1.54× |
| Table 2 | psi | 1 | 0.79× | 0.79× | 0.79× |
| Table 3 | hybit | 1 | 0.68× | 0.68× | 0.68× |
| Table 3 | pluck | 1 | 1.53× | 1.53× | 1.53× |
| Table 3 | psi | 1 | 0.93× | 0.93× | 0.93× |
| Table 3 | siren | 1 | 3.78× | 3.78× | 3.78× |
| Table 3 | slicstan | 1 | 0.97× | 0.97× | 0.97× |
| Table 3 | turing | 1 | 0.58× | 0.58× | 0.58× |
| Table 5 | t_posterior | 1 | 1.00× | 1.00× | 1.00× |
| Table 5 | t_symbolic | 2 | 1.84× | 1.69× | 2.00× |
| Table 5 | t_wmc | 1 | 0.90× | 0.90× | 0.90× |

## Figure 1

Compare the panels in `fig1/` with the paper's in `experiments/reference/fig1/`. The fonts differ (the image has no Linux Libertine).

| Program | LoC paper | LoC here | Time paper (s) | Time here (s) | PSI here |
|---|---|---|---|---|---|
| fig_changepoint_listing | 40 | 40 | 0.9 | 1.4 | t/o |
| fig_dawid_skene_listing | 39 | 39 | 0.016 | 0.016 | t/o |
| fig_mixture_listing | 32 | 32 | 1.4 | 1.1 | t/o |
| fig_polyreg_listing | 35 | 35 | 1.3 | 1.5 | t/o |
| fig_sir_listing | 52 | 52 | 0.9 | 0.9 | t/o |

## Every cell of Tables 2 and 3

| Table | Program | System | Paper | This run | This run / paper |
|---|---|---|---|---|---|
| Table 2 | burglary_alarm | pluck | 1.26e-04 | n/r |  |
| Table 2 | burglary_alarm | psi | 0.0883 | n/r |  |
| Table 2 | burglary_alarm | hybit | 2.56e-03 | n/r |  |
| Table 2 | diamond_network | pluck | 0.0121 | n/r |  |
| Table 2 | diamond_network | psi | 1.9 | n/r |  |
| Table 2 | diamond_network | hybit | 9.08e-03 | n/r |  |
| Table 2 | typos | pluck | 0.0166 | n/r |  |
| Table 2 | addFun_sum | pluck | 6.54e-05 | n/r |  |
| Table 2 | addFun_sum | psi | 0.017 | n/r |  |
| Table 2 | addFun_sum | hybit | 0.311 | n/r |  |
| Table 2 | conjugate_gaussians | pluck | 1.21e-04 | n/r |  |
| Table 2 | conjugate_gaussians | psi | 0.018 | n/r |  |
| Table 2 | conjugate_gaussians | hybit | 1.07 | n/r |  |
| Table 2 | linear_regression | pluck | 4.15e-04 | n/r |  |
| Table 2 | linear_regression | psi | 0.632 | n/r |  |
| Table 2 | linear_regression | hybit | 3.19 | n/r |  |
| Table 2 | clickGraph | pluck | 5.64e-04 | n/r |  |
| Table 2 | clickGraph | psi | 3.22 | n/r |  |
| Table 2 | clickGraph | hybit | 0.48 | n/r |  |
| Table 2 | clinicalTrial1 | pluck | 0.0359 | n/r |  |
| Table 2 | clinicalTrial1 | psi | t/o | n/r |  |
| Table 2 | clinicalTrial1 | hybit | 3.83 | n/r |  |
| Table 2 | clinicalTrial2 | pluck | 2.80e-04 | n/r |  |
| Table 2 | clinicalTrial2 | psi | 0.438 | n/r |  |
| Table 2 | clinicalTrial2 | hybit | 0.241 | n/r |  |
| Table 2 | coinBias | pluck | 1.72e-04 | 2.65e-04 | 1.54× |
| Table 2 | coinBias | psi | 0.201 | 0.159 | 0.79× |
| Table 2 | coinBias | hybit | 0.0449 | 0.0333 | 0.74× |
| Table 2 | coinBias | hybit_err | 2.00e-03 | 2.00e-03 | 1.00× |
| Table 2 | GPA | pluck | 1.64e-03 | n/r |  |
| Table 2 | GPA | psi | 0.915 | n/r |  |
| Table 2 | GPA | hybit | 0.171 | n/r |  |
| Table 2 | normal_mixture | pluck | 2.89 | n/r |  |
| Table 2 | normal_mixture | psi | t/o | n/r |  |
| Table 2 | normal_mixture | hybit | 43.5 | n/r |  |
| Table 2 | addFun_max | psi | 0.0209 | n/r |  |
| Table 2 | addFun_max | hybit | 514 | n/r |  |
| Table 2 | spacex | psi | 0.102 | n/r |  |
| Table 2 | spacex | hybit | 21.1 | n/r |  |
| Table 2 | trueskill | psi | 0.129 | n/r |  |
| Table 2 | trueskill | hybit | 116 | n/r |  |
| Table 2 | tug_of_war | psi | 11.5 | n/r |  |
| Table 2 | tug_of_war | hybit | 583 | n/r |  |
| Table 2 | weekend | psi | 0.0745 | n/r |  |
| Table 2 | weekend | hybit | 0.149 | n/r |  |
| Table 3 | spike_and_slab | pluck | 6.73e-03 | n/r |  |
| Table 3 | spike_and_slab | psi | 21.1 | n/r |  |
| Table 3 | spike_and_slab | hybit | t/o | n/r |  |
| Table 3 | spike_and_slab | slicstan | n/i | n/r |  |
| Table 3 | spike_and_slab | siren | t/o | n/r |  |
| Table 3 | spike_and_slab | turing | r/e | n/r |  |
| Table 3 | polyreg | pluck | 0.0105 | n/r |  |
| Table 3 | polyreg | psi | 111 | n/r |  |
| Table 3 | polyreg | hybit | t/o | n/r |  |
| Table 3 | polyreg | slicstan | r/e | n/r |  |
| Table 3 | polyreg | siren | t/o | n/r |  |
| Table 3 | polyreg | turing | r/e | n/r |  |
| Table 3 | run_walk_hrm | pluck | 6.15e-03 | n/r |  |
| Table 3 | run_walk_hrm | psi | t/o | n/r |  |
| Table 3 | run_walk_hrm | hybit | t/o | n/r |  |
| Table 3 | run_walk_hrm | slicstan | r/e | n/r |  |
| Table 3 | run_walk_hrm | siren | 0.0699 | n/r |  |
| Table 3 | run_walk_hrm | turing | 1.06 | n/r |  |
| Table 3 | sir_model_n12 | pluck | 0.0105 | n/r |  |
| Table 3 | sir_model_n12 | psi | t/o | n/r |  |
| Table 3 | sir_model_n12 | hybit | r/e | n/r |  |
| Table 3 | sir_model_n12 | slicstan | 10.4 | n/r |  |
| Table 3 | sir_model_n12 | siren | 6.4 | n/r |  |
| Table 3 | sir_model_n12 | turing | 14.3 | n/r |  |
| Table 3 | sir_model_n72 | pluck | 121 | n/r |  |
| Table 3 | sir_model_n72 | psi | t/o | n/r |  |
| Table 3 | sir_model_n72 | hybit | t/o | n/r |  |
| Table 3 | sir_model_n72 | slicstan | t/o | n/r |  |
| Table 3 | sir_model_n72 | siren | 210 | n/r |  |
| Table 3 | sir_model_n72 | turing | r/e | n/r |  |
| Table 3 | f81_rate | pluck | 0.092 | n/r |  |
| Table 3 | f81_rate | psi | t/o | n/r |  |
| Table 3 | f81_rate | hybit | 0.0619 | n/r |  |
| Table 3 | f81_rate | slicstan | n/i | n/r |  |
| Table 3 | f81_rate | turing | r/e | n/r |  |
| Table 3 | f81_joint | pluck | 0.907 | n/r |  |
| Table 3 | f81_joint | psi | t/o | n/r |  |
| Table 3 | f81_joint | turing | r/e | n/r |  |
| Table 3 | pcfg_ambiguous | pluck | 0.276 | n/r |  |
| Table 3 | pcfg_ambiguous | turing | t/o | n/r |  |
| Table 3 | pcfg_n020 | pluck | 0.25 | n/r |  |
| Table 3 | pcfg_n020 | turing | 37.4 | n/r |  |
| Table 3 | changepoint_coal_k1 | pluck | 0.0144 | 0.0221 | 1.53× |
| Table 3 | changepoint_coal_k1 | psi | 17.6 | 16.3 | 0.93× |
| Table 3 | changepoint_coal_k1 | hybit | 2.59 | 1.76 | 0.68× |
| Table 3 | changepoint_coal_k1 | slicstan | 2.88 | 2.79 | 0.97× |
| Table 3 | changepoint_coal_k1 | siren | 0.0733 | 0.277 | 3.78× |
| Table 3 | changepoint_coal_k1 | turing | 2.32 | 1.35 | 0.58× |
| Table 3 | changepoint_coal_n020 | pluck | 0.684 | n/r |  |
| Table 3 | changepoint_coal_n020 | psi | t/o | n/r |  |
| Table 3 | changepoint_coal_n020 | hybit | r/e | n/r |  |
| Table 3 | changepoint_coal_n020 | siren | 18.6 | n/r |  |
| Table 3 | changepoint_coal_n020 | turing | r/e | n/r |  |
| Table 3 | changepoint_coal_n050 | pluck | 178 | n/r |  |
| Table 3 | changepoint_coal_n050 | psi | t/o | n/r |  |
| Table 3 | changepoint_coal_n050 | hybit | t/o | n/r |  |
| Table 3 | changepoint_coal_n050 | siren | 8.26 | n/r |  |
| Table 3 | changepoint_coal_n050 | turing | 575 | n/r |  |
| Table 3 | dawid_skene_n08 | pluck | 3.28e-03 | n/r |  |
| Table 3 | dawid_skene_n08 | psi | t/o | n/r |  |
| Table 3 | dawid_skene_n08 | hybit | t/o | n/r |  |
| Table 3 | dawid_skene_n08 | slicstan | 27.8 | n/r |  |
| Table 3 | dawid_skene_n08 | siren | 0.217 | n/r |  |
| Table 3 | dawid_skene_n08 | turing | 15.1 | n/r |  |
| Table 3 | dawid_skene_n16 | pluck | 0.0612 | n/r |  |
| Table 3 | dawid_skene_n16 | psi | t/o | n/r |  |
| Table 3 | dawid_skene_n16 | hybit | t/o | n/r |  |
| Table 3 | dawid_skene_n16 | slicstan | t/o | n/r |  |
| Table 3 | dawid_skene_n16 | siren | 6.47 | n/r |  |
| Table 3 | dawid_skene_n16 | turing | r/e | n/r |  |
| Table 3 | hmm_first_order_n02 | pluck | 5.17e-04 | n/r |  |
| Table 3 | hmm_first_order_n02 | psi | 0.644 | n/r |  |
| Table 3 | hmm_first_order_n02 | hybit | r/e | n/r |  |
| Table 3 | hmm_first_order_n02 | slicstan | 0.346 | n/r |  |
| Table 3 | hmm_first_order_n02 | siren | 0.0257 | n/r |  |
| Table 3 | hmm_first_order_n02 | turing | 1.39 | n/r |  |
| Table 3 | hmm_first_order_n08 | pluck | 0.0418 | n/r |  |
| Table 3 | hmm_first_order_n08 | psi | t/o | n/r |  |
| Table 3 | hmm_first_order_n08 | hybit | r/e | n/r |  |
| Table 3 | hmm_first_order_n08 | slicstan | 3.24 | n/r |  |
| Table 3 | hmm_first_order_n08 | siren | 0.885 | n/r |  |
| Table 3 | hmm_first_order_n08 | turing | 31.2 | n/r |  |
| Table 3 | hmm_first_order_n14 | pluck | 43.8 | n/r |  |
| Table 3 | hmm_first_order_n14 | psi | t/o | n/r |  |
| Table 3 | hmm_first_order_n14 | hybit | t/o | n/r |  |
| Table 3 | hmm_first_order_n14 | slicstan | 17.8 | n/r |  |
| Table 3 | hmm_first_order_n14 | siren | 0.81 | n/r |  |
| Table 3 | hmm_first_order_n14 | turing | 10 | n/r |  |
| Table 3 | hmm_second_order_n04 | pluck | 1.72e-03 | n/r |  |
| Table 3 | hmm_second_order_n04 | psi | 35.6 | n/r |  |
| Table 3 | hmm_second_order_n04 | hybit | 77 | n/r |  |
| Table 3 | hmm_second_order_n04 | siren | 0.0675 | n/r |  |
| Table 3 | hmm_second_order_n04 | turing | 1.03 | n/r |  |
| Table 3 | hmm_second_order_n08 | pluck | 0.0473 | n/r |  |
| Table 3 | hmm_second_order_n08 | psi | t/o | n/r |  |
| Table 3 | hmm_second_order_n08 | hybit | t/o | n/r |  |
| Table 3 | hmm_second_order_n08 | siren | 0.478 | n/r |  |
| Table 3 | hmm_second_order_n08 | turing | 30.2 | n/r |  |
| Table 3 | hmm_second_order_n14 | pluck | 47.1 | n/r |  |
| Table 3 | hmm_second_order_n14 | psi | t/o | n/r |  |
| Table 3 | hmm_second_order_n14 | hybit | t/o | n/r |  |
| Table 3 | hmm_second_order_n14 | siren | 0.482 | n/r |  |
| Table 3 | hmm_second_order_n14 | turing | 10.3 | n/r |  |
| Table 3 | hmm_factorial_n02 | pluck | 1.71e-03 | n/r |  |
| Table 3 | hmm_factorial_n02 | psi | 2.23 | n/r |  |
| Table 3 | hmm_factorial_n02 | hybit | t/o | n/r |  |
| Table 3 | hmm_factorial_n02 | siren | 0.043 | n/r |  |
| Table 3 | hmm_factorial_n02 | turing | 1.19 | n/r |  |
| Table 3 | hmm_factorial_n04 | pluck | 0.0142 | n/r |  |
| Table 3 | hmm_factorial_n04 | psi | t/o | n/r |  |
| Table 3 | hmm_factorial_n04 | hybit | t/o | n/r |  |
| Table 3 | hmm_factorial_n04 | siren | 0.142 | n/r |  |
| Table 3 | hmm_factorial_n04 | turing | 3.52 | n/r |  |
| Table 3 | hmm_factorial_n06 | pluck | 0.37 | n/r |  |
| Table 3 | hmm_factorial_n06 | psi | t/o | n/r |  |
| Table 3 | hmm_factorial_n06 | hybit | t/o | n/r |  |
| Table 3 | hmm_factorial_n06 | siren | 0.102 | n/r |  |
| Table 3 | hmm_factorial_n06 | turing | 4.35 | n/r |  |
| Table 3 | king_ei_d60 | pluck | 0.329 | n/r |  |
| Table 3 | king_ei_d60 | psi | t/o | n/r |  |
| Table 3 | king_ei_d60 | hybit | 137 | n/r |  |
| Table 3 | king_ei_d60 | siren | 3.73 | n/r |  |
| Table 3 | king_ei_d60 | turing | r/e | n/r |  |
| Table 3 | king_ei_d20 | pluck | 22.6 | n/r |  |
| Table 3 | king_ei_d20 | psi | t/o | n/r |  |
| Table 3 | king_ei_d20 | hybit | t/o | n/r |  |
| Table 3 | king_ei_d20 | siren | 4.34 | n/r |  |
| Table 3 | king_ei_d20 | turing | r/e | n/r |  |
| Table 3 | outliers_n08 | pluck | 6.56e-03 | n/r |  |
| Table 3 | outliers_n08 | psi | 141 | n/r |  |
| Table 3 | outliers_n08 | hybit | r/e | n/r |  |
| Table 3 | outliers_n08 | slicstan | 3.25 | n/r |  |
| Table 3 | outliers_n08 | siren | 0.241 | n/r |  |
| Table 3 | outliers_n08 | turing | 13.4 | n/r |  |
| Table 3 | outliers_n18 | pluck | 18.4 | n/r |  |
| Table 3 | outliers_n18 | psi | t/o | n/r |  |
| Table 3 | outliers_n18 | hybit | t/o | n/r |  |
| Table 3 | outliers_n18 | slicstan | 39.2 | n/r |  |
| Table 3 | outliers_n18 | siren | 1.09 | n/r |  |
| Table 3 | outliers_n18 | turing | 25.4 | n/r |  |
| Table 3 | soft_kmeans_n04 | pluck | 3.30e-03 | n/r |  |
| Table 3 | soft_kmeans_n04 | psi | 18 | n/r |  |
| Table 3 | soft_kmeans_n04 | hybit | t/o | n/r |  |
| Table 3 | soft_kmeans_n04 | slicstan | r/e | n/r |  |
| Table 3 | soft_kmeans_n04 | siren | 0.0473 | n/r |  |
| Table 3 | soft_kmeans_n04 | turing | r/e | n/r |  |
| Table 3 | soft_kmeans_n12 | pluck | 21.1 | n/r |  |
| Table 3 | soft_kmeans_n12 | psi | t/o | n/r |  |
| Table 3 | soft_kmeans_n12 | hybit | t/o | n/r |  |
| Table 3 | soft_kmeans_n12 | slicstan | r/e | n/r |  |
| Table 3 | soft_kmeans_n12 | siren | 0.14 | n/r |  |
| Table 3 | soft_kmeans_n12 | turing | r/e | n/r |  |
| Table 3 | secret_ballot_n04 | pluck | 1.33e-03 | n/r |  |
| Table 3 | secret_ballot_n04 | psi | 215 | n/r |  |
| Table 3 | secret_ballot_n04 | hybit | t/o | n/r |  |
| Table 3 | secret_ballot_n04 | siren | 0.241 | n/r |  |
| Table 3 | secret_ballot_n04 | turing | 1.72 | n/r |  |
| Table 3 | secret_ballot_n128 | pluck | 71.2 | n/r |  |
| Table 3 | secret_ballot_n128 | psi | t/o | n/r |  |
| Table 3 | secret_ballot_n128 | hybit | t/o | n/r |  |
| Table 3 | secret_ballot_n128 | siren | 328 | n/r |  |
| Table 3 | secret_ballot_n128 | turing | r/e | n/r |  |
| Table 3 | soccer_n03 | pluck | 4.95e-03 | n/r |  |
| Table 3 | soccer_n03 | psi | t/o | n/r |  |
| Table 3 | soccer_n03 | turing | 11.8 | n/r |  |
| Table 3 | soccer_n07 | pluck | 7.1 | n/r |  |
| Table 3 | soccer_n07 | psi | t/o | n/r |  |
| Table 3 | soccer_n07 | turing | r/e | n/r |  |
