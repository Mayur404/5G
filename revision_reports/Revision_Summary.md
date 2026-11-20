# CARE — Deep Review revision summary

Full Deep Review comments for each round are in `revised1/review1.md` and `revised2/review2.md`. The PDF that was reviewed in each round is next to its review (`5G.pdf`), with the LaTeX it was built from (`5G_source.tex`). The final paper is `paper/main.pdf`.

## Round 1 (review of `revised1/5G.pdf`) → Revision 1

| Review comment | Change made |
|---|---|
| Reward $R_j$, objective $J_\lambda$, latency and QoS equations missing; `(??)` cross-references | Restored Eqs. for $R_j$, $J_\lambda$, latency $\ell$ and admission; all references resolve |
| Leftover drafting text ("must be consistent with the simulator's load calculation"), spliced Bellman sentence | Removed; $\rho_c=(\sum a_u+h_c)/C$ defined explicitly |
| Abstract/contributions said "exactly the RBs it displaced" and admission-only credit | Reworded: admission **or** full-demand QoS rescue; charged a budget-balanced proportional share |
| Counterfactual free capacity, radio failure, outage dynamics, slice partition unspecified | Added $F_c^{\mathrm{cf}}=F_c+\max(0,m-u_c)$, outage behaviour, same-interval ordering, slice probabilities (0.42, 0.28, 0.30) |
| Retiming rule, $H_c/P_c$ notation, mask updates, legal actions in Bellman max | Retiming rule written out; $H_c,P_c$ defined as RB sums; UE-ID order and mask update stated; inner max restricted to legal actions |
| Replay buffer reward non-stationarity under multi-UE coupling | Acknowledged in Sec. VI-C |
| Double Q / conservative offline RL not discussed | Added rationale for inference-time ensemble; left as future work |
| PPO diagnosis confounds (critic, 9:1 action ratio, 160 vs 240 episodes) | Stated that PPO uses a value baseline, the 9:1 ratio, and that both variants converge early |
| $\tau$ grid, second Holm family, "outage length" feature, calibration intercept | $\tau\in\{0,0.6,0.8,0.9,0.95\}$ plus never-hold; family described; feature is $T_o/10$; intercept unpenalised, $1/N$ loss |
| "Statistically indistinguishable" wording | Replaced: unadjusted CIs exclude zero; not significant after Holm; full-service gap significant but negligible |
| Hold-share non-monotonicity, gap reference, slice order / "of all decisions" | Explained; gaps labelled vs global tuned rule; slice order eMBB/uRLLC/mMTC |
| "Unprotected activation" terminology | Renamed "free-RB-only activation" and defined |
| Prep-delay sensitivity mechanism | Linked to CARE's shorter commitment lead (5.22 vs 7.30 intervals) |
| PPO on-policy wait shown only at λ=1,16 in Table VI / Fig. 6 | Captions state it was run only at those prices |
| Guard pools / shadow-cluster baselines | Exclusion justified in Sec. II; future work |
| ASCII minus signs, ref [12] page spacing, ref [17] umlaut | Fixed |
| Repository URL | Stated as supplementary artifact (submission zip) |

## Round 2 (review of `revised2/5G.pdf`) → Revision 2

| Review comment | Change made |
|---|---|
| Headline numbers lack baseline/price context; QoS trade-off hidden; non-monotone hold share | Abstract now states λ=1, no-reservation and reactive baselines, floor holds trade full-demand QoS, and "between the lowest and highest prices" |
| Problem scenario gives only one side of the timing trade-off | Added the late-commit / late-activation risks |
| Latency gate can never bind when full demand is met | Stated explicitly ($\ell\le1.75\,\ell_s$ below every limit; QoS reduces to admitted with full demand) |
| Asymmetry between per-contract rescue credit and shared displacement | Explained and justified after Proposition 1 |
| $D_j$ overloaded (per-interval vs cumulative) | $D_j$ defined as summed over the contract's active intervals (matches code) |
| $\mathcal{U}_c, a_u$ used before definition | Forward reference added |
| Fig. 1 hardcodes $t+1$ | Caption states $\delta_p=1$ |
| Algorithm 1 hat accent, log-loss sign, slice one-hot size | Fixed / defined |
| Pooled probability-scaled baseline, Double-Q / sliding-window replay, hierarchical PPO head, realistic A3 mobility, contract-mechanics full metrics, prep-delay observability, censoring, Holm family size | Not addressed: require new experiments; noted as limitations / future work |
