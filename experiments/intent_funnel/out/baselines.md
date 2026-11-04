# Baselines (CPU)

Full prompt = 3730 DeepSeek tokens (wiki + 16 tool schemas). Per-set: safe = needed write tools ⊆ route or fallback; risk = misroutes among routed; token = routed/full prompt.

## Keyword rules (no hit -> continue)

**replay on 249 real test trajectories (thresholds cross-fit, picks [(0.05, 0.0), (0.05, 0.0)]):** write coverage 93.8%, task coverage 91.2%, step token 0.56, trajectories that fell back 0.0%

Per-set rows use t_include=0.05, gate=0.0.

| set | n | coverage | risk | token | top1 | macroF1 | ECE |
|---|---|---|---|---|---|---|---|
| tau_natural | 111 | 100.0% | 9.9% | 0.61 | – | – | – |
| tau_terse | 4 | 100.0% | 0.0% | 0.69 | – | – | – |
| tau_traj | 1597 | 100.0% | 6.3% | 0.44 | – | – | – |
| abcd | 830 | 100.0% | 4.6% | 0.37 | 0.130 | 0.325 | 0.870 |
| bitext | 5164 | 100.0% | 8.3% | 0.37 | 0.175 | 0.282 | 0.825 |
| clinc_oos | 1000 | 100.0% | 0.0% | 0.36 | 0.000 | 0.000 | 1.000 |
| banking_id_oos | 1076 | 100.0% | 0.0% | 0.37 | 0.000 | 0.000 | 1.000 |

## TF-IDF + LR — A_no_traj_sent (n_train=14855)

**replay on 249 real test trajectories (thresholds cross-fit, picks [(0.1, 0.4), (0.25, 0.0)]):** write coverage 89.0%, task coverage 84.7%, step token 0.68, trajectories that fell back 31.7%

Per-set rows use t_include=0.1, gate=0.4.

| set | n | coverage | risk | token | top1 | macroF1 | ECE |
|---|---|---|---|---|---|---|---|
| tau_natural | 111 | 95.5% | 17.9% | 0.61 | – | – | – |
| tau_terse | 4 | 100.0% | 0.0% | 0.70 | – | – | – |
| tau_traj | 1597 | 87.9% | 7.9% | 0.53 | – | – | – |
| abcd | 830 | 98.4% | 0.5% | 0.43 | 0.872 | 0.662 | 0.065 |
| bitext | 5164 | 99.9% | 0.0% | 0.39 | 0.997 | 0.997 | 0.012 |
| clinc_oos | 1000 | 97.1% | 0.0% | 0.39 | 0.920 | 0.958 | 0.120 |
| banking_id_oos | 1076 | 99.3% | 0.0% | 0.38 | 0.969 | 0.984 | 0.035 |

## TF-IDF + LR — D_with_traj_whole_nogroups (n_train=28650)

**replay on 249 real test trajectories (thresholds cross-fit, picks [(0.15, 0.0), (0.25, 0.0)]):** write coverage 95.7%, task coverage 94.4%, step token 0.56, trajectories that fell back 17.3%

Per-set rows use t_include=0.15, gate=0.0.

| set | n | coverage | risk | token | top1 | macroF1 | ECE |
|---|---|---|---|---|---|---|---|
| tau_natural | 111 | 93.7% | 33.7% | 0.51 | – | – | – |
| tau_terse | 4 | 100.0% | 25.0% | 0.54 | – | – | – |
| tau_traj | 1597 | 98.4% | 7.0% | 0.43 | – | – | – |
| abcd | 830 | 100.0% | 0.7% | 0.39 | 0.908 | 0.717 | 0.039 |
| bitext | 5164 | 100.0% | 0.0% | 0.38 | 0.997 | 0.996 | 0.008 |
| clinc_oos | 1000 | 99.8% | 0.0% | 0.36 | 0.869 | 0.930 | 0.129 |
| banking_id_oos | 1076 | 100.0% | 0.0% | 0.36 | 0.949 | 0.974 | 0.033 |

## TF-IDF + LR — D_with_traj_whole (n_train=28650)

**replay on 249 real test trajectories (thresholds cross-fit, picks [(0.15, 0.0), (0.15, 0.0)]):** write coverage 98.3%, task coverage 97.6%, step token 0.60, trajectories that fell back 10.0%

Per-set rows use t_include=0.15, gate=0.0.

| set | n | coverage | risk | token | top1 | macroF1 | ECE |
|---|---|---|---|---|---|---|---|
| tau_natural | 111 | 93.7% | 29.8% | 0.57 | – | – | – |
| tau_terse | 4 | 100.0% | 25.0% | 0.64 | – | – | – |
| tau_traj | 1597 | 98.4% | 4.6% | 0.46 | – | – | – |
| abcd | 830 | 100.0% | 0.7% | 0.39 | 0.908 | 0.717 | 0.039 |
| bitext | 5164 | 100.0% | 0.0% | 0.39 | 0.997 | 0.996 | 0.008 |
| clinc_oos | 1000 | 99.8% | 0.0% | 0.37 | 0.869 | 0.930 | 0.129 |
| banking_id_oos | 1076 | 100.0% | 0.0% | 0.37 | 0.949 | 0.974 | 0.033 |

## TF-IDF + LR — D_with_traj_sent (n_train=28650)

**replay on 249 real test trajectories (thresholds cross-fit, picks [(0.4, 0.0), (0.4, 0.0)]):** write coverage 96.7%, task coverage 96.4%, step token 0.59, trajectories that fell back 10.8%

Per-set rows use t_include=0.4, gate=0.0.

| set | n | coverage | risk | token | top1 | macroF1 | ECE |
|---|---|---|---|---|---|---|---|
| tau_natural | 111 | 100.0% | 20.7% | 0.57 | – | – | – |
| tau_terse | 4 | 100.0% | 0.0% | 0.67 | – | – | – |
| tau_traj | 1597 | 98.2% | 5.2% | 0.45 | – | – | – |
| abcd | 830 | 98.6% | 1.0% | 0.39 | 0.819 | 0.662 | 0.080 |
| bitext | 5164 | 99.9% | 0.0% | 0.38 | 0.997 | 0.996 | 0.008 |
| clinc_oos | 1000 | 91.5% | 0.0% | 0.41 | 0.868 | 0.929 | 0.128 |
| banking_id_oos | 1076 | 99.3% | 0.0% | 0.37 | 0.928 | 0.963 | 0.039 |

Latency per request (cpu): p50 2.7 ms, p95 3.1 ms

## TF-IDF + LR — E_traj_only_whole (n_train=13795)

**replay on 249 real test trajectories (thresholds cross-fit, picks [(0.15, 0.0), (0.15, 0.0)]):** write coverage 99.4%, task coverage 99.2%, step token 0.62, trajectories that fell back 3.6%

Per-set rows use t_include=0.15, gate=0.0.

| set | n | coverage | risk | token | top1 | macroF1 | ECE |
|---|---|---|---|---|---|---|---|
| tau_natural | 111 | 98.2% | 27.5% | 0.58 | – | – | – |
| tau_terse | 4 | 100.0% | 25.0% | 0.62 | – | – | – |
| tau_traj | 1597 | 99.4% | 2.9% | 0.47 | – | – | – |
| abcd | 830 | 98.8% | 5.0% | 0.45 | 0.067 | 0.207 | 0.607 |
| bitext | 5164 | 97.6% | 5.0% | 0.45 | 0.084 | 0.196 | 0.586 |
| clinc_oos | 1000 | 99.4% | 0.0% | 0.41 | 0.000 | 0.000 | 0.807 |
| banking_id_oos | 1076 | 99.8% | 0.0% | 0.39 | 0.000 | 0.000 | 0.816 |
