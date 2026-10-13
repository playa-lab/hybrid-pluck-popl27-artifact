# Comparison with the paper: `large_king_ei_d15`

Tier `paper` on x86_64 with 64 CPUs; timeouts: pluck 600 s, psi 600 s, hybit 600 s, slicstan 600 s, siren 600 s, turing 600 s.

## Summary

- Claims: 3 of 3 checked claims agree.
- Tables 4 and 5: 0 counts differ from the paper, and 0 SPN sizes that vary between runs; 96 programs have no counts in this run.

## The paper's claims

| Claim | Paper reference | Paper | This run | Agrees |
|---|---|---|---|---|
| Tables 4 and 5's BDD and SPN sizes match exactly (except the SPN sizes in lists/spn_varies.txt) | Tables 4, 5 | 5 cells | 5 match | yes |
| The overview example's numbers | §2.4 | Z 4.9e-03; P(s2) 0.57; iota 0.143 -> 0.178; rho 0.333 -> 0.329 | Z 4.9e-03; P(s2) 0.57; iota 0.143 -> 0.178; rho 0.333 -> 0.329 | yes |
| This run's Pluck answers equal the fixed exact answers (largest relative difference) | reference/exact_answers.csv | 0 | 0.0e+00 over 1 programs | yes |

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

Programs with no counts in this run: GPA, addFun_sum, burglary_alarm, changepoint_coal_k1, changepoint_coal_n010, changepoint_coal_n015, changepoint_coal_n020, changepoint_coal_n025, changepoint_coal_n030, changepoint_coal_n035, changepoint_coal_n040, changepoint_coal_n050, clickGraph, clinicalTrial1, clinicalTrial2, coinBias, conjugate_gaussians, dawid_skene_n08, dawid_skene_n12, dawid_skene_n16, dawid_skene_n18, dawid_skene_n20, dawid_skene_n22, diamond_network, f81_joint, f81_rate, hmm_factorial_n02, hmm_factorial_n03, hmm_factorial_n04, hmm_factorial_n05, hmm_factorial_n06, hmm_factorial_n07, hmm_first_order_n02, hmm_first_order_n04, hmm_first_order_n06, hmm_first_order_n08, hmm_first_order_n10, hmm_first_order_n12, hmm_first_order_n14, hmm_second_order_n02, hmm_second_order_n04, hmm_second_order_n06, hmm_second_order_n08, hmm_second_order_n10, hmm_second_order_n12, hmm_second_order_n14, king_ei_d20, king_ei_d25, king_ei_d30, king_ei_d40, king_ei_d60, king_ei_d80, linear_regression, normal_mixture, outliers_n08, outliers_n10, outliers_n12, outliers_n14, outliers_n16, outliers_n18, pcfg_ambiguous, pcfg_n010, pcfg_n020, pcfg_n040, pcfg_n060, pcfg_n080, pcfg_n100, polyreg, run_walk_hrm, secret_ballot_n04, secret_ballot_n08, secret_ballot_n128, secret_ballot_n16, secret_ballot_n32, secret_ballot_n64, sir_model_n12, sir_model_n24, sir_model_n36, sir_model_n48, sir_model_n60, sir_model_n72, soccer_n03, soccer_n04, soccer_n05, soccer_n06, soccer_n07, soccer_n08, soft_kmeans_n02, soft_kmeans_n04, soft_kmeans_n06, soft_kmeans_n08, soft_kmeans_n10, soft_kmeans_n11, soft_kmeans_n12, spike_and_slab, typos.

### Tables 4 and 5: SPN sizes that vary between runs

The programs in `lists/spn_varies.txt` build SPNs of different sizes on different runs or machines; their BDD counts and answers do not change. Not a disagreement.

None.

## Timing: this run / paper

Geometric mean, smallest and largest ratio of this run's time to the paper's, over the cells where both have a time. Approximate systems' times are those of the budget each run selected.

| Table | Metric | Cells | Geometric mean | Min | Max |
|---|---|---|---|---|---|
| Table 5 | t_posterior | 1 | 0.63× | 0.63× | 0.63× |
| Table 5 | t_symbolic | 1 | 0.72× | 0.72× | 0.72× |
| Table 5 | t_wmc | 1 | 0.76× | 0.76× | 0.76× |
