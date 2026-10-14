# Mathematical specification: predictive handover reservation in a sliced 5G RAN

> Implementation status: this is the source-based proposed specification. The runnable artifact uses a declared synthetic predictor and priority scheduler rather than every proposed calibrated-error, throughput, and slice-floor interface below. See [SIMULATION_REVIEW.md](SIMULATION_REVIEW.md) for the implemented equations, verified results, and remaining differences.

## Scope and provenance

This document uses all three supplied files: [text1.txt](text1.txt) (problem statement), [text2.txt](text2.txt) (initial formulation), and [*PreNS: A Hybrid Predictive and Real-Time Resource Allocation Framework for 5G and Beyond RAN Network Slicing*](<PreNS_A_Hybrid_Predictive_and_Real-Time_Resource_Allocation_Framework_for_5G_and_Beyond_RAN_Network_Slicing (1).pdf>) (supplied PDF, abbreviated **P**). The PPO and generalized advantage equations come from the original [PPO paper](https://arxiv.org/pdf/1707.06347) (**O**) and [GAE paper](https://arxiv.org/pdf/1506.02438) (**G**). **N** means an equation explicitly in `text2.txt`; **D** is an algebraic/accounting consequence of cited equations; **S** is an explicit proposed system definition motivated by the two notes. A source equation does not automatically become a recommended implementation equation.

The paper predicts **per-slice traffic** and controls **slice allocation with DDQN**. The notes request **per-UE target-cell reservation with PPO**. There is no supplied mobility predictor, demand-to-RB conversion, handover simulator, calibrated probability model, or numerical reward weights. The equations below define the reservation controller conditional on externally supplied predictions and measured outcomes. Section “Unspecified inputs” identifies the remaining contracts; claiming a raw-telemetry-to-trained-controller implementation from these sources alone would require inventing formulas.

All resource counts below are **integer RBs available in one common scheduling interval** of duration \(\Delta\). Candidate cells have positive capacity \(K_j>0\). Reservations block capacity during each interval in which they are active. Throughput is in bit/s, delay and time in seconds, probabilities and reward components are dimensionless. Never add RBs to bit/s or average quantities with different units.

## 1. System model

Let \(i\in\mathcal U\) index UEs, \(j\in\mathcal C\) cells, \(s(i)\in\mathcal S\) the slice of UE \(i\), and \(t=0,1,\ldots\) decision intervals. The source cell of \(i\) is \(c_{i,t}\); a candidate destination is \(j\ne c_{i,t}\). \(K_j\) is cell \(j\)'s RB capacity per interval, \(A_{j,t}\) RBs already allocated to served UEs, and \(B_{i,j,t}\) RBs actively reserved for UE \(i\) at \(j\). **D, from P's allocation bounds and N's capacity rule:**

**(1)** \(A_{j,t}+\sum_i B_{i,j,t}\le K_j,\qquad A_{j,t},B_{i,j,t},K_j\in\mathbb Z_{\ge0}.\)

The directly observable uncommitted capacity is **D**:

**(2)** \(F_{j,t}=K_j-A_{j,t}-\sum_i B_{i,j,t}\ge0.\)

\(F_{j,t}\) is free RBs, not total remaining capacity including other reservations. If existing slice allocations \(A_{j,s,t}\) and reservation counts \(B_{j,s,t}=\sum_{i:s(i)=s}B_{i,j,t}\) are exposed, a per-slice floor \(M_{j,s,t}\) can be checked with **S**, motivated by N's slice isolation requirement:

**(3)** \(A_{j,s,t}+B_{j,s,t}\ge M_{j,s,t}\quad\text{for each slice whose floor is enforceable at }(j,t).\)

This is a *floor on committed slice RBs*, not proof of a throughput/latency SLA. When floors apply to *currently served* users, reservations do not satisfy them; then use \(A_{j,s,t}\ge M_{j,s,t}\). Floors are feasible only if their sum does not exceed cell capacity, and an initial floor violation cannot be repaired by masking a no-op reservation action. Treat (3) as a monitored scheduler constraint, not a guarantee by the PPO controller. The floor semantics must be set by the simulator/operator before use. Capacity (1) applies in either case.

**P, Sec. III, eqs. (1)–(4), transcribed with new non-conflicting notation:** for served UE \(i\), achieved throughput \(T_{i,t}\), required minimum \(T_i^{\min}\), achieved latency \(L_{i,t}\), slice maximum \(L_{s(i)}^{\max}\), slice allocation \(A_{j,s,t}\), and UE allocation \(A_{i,j,t}\),

**(4)** \(T_{i,t}\ge T_i^{\min},\qquad L_{i,t}\le L_{s(i)}^{\max}.\)

**(5)** \(\sum_s A_{j,s,t}\le K_j,\qquad \rho_{j,s,t}=A_{j,s,t}/K_j\quad(K_j>0).\)

**(6)** \(\sum_{i:c_{i,t}=j,\,s(i)=s}A_{i,j,t}\le A_{j,s,t},\qquad \rho_{i,j,t}=A_{i,j,t}/A_{j,s,t}\quad(A_{j,s,t}>0).\)

\(\rho\)'s are allocation fractions. Equations (4)–(6) describe PreNS background allocation; the reservation controller operates on (1)–(3). The paper does not give a physical mapping from RBs to \(T_{i,t}\) or \(L_{i,t}\); these must be measured or supplied by the environment.

## 2. Prediction

At time \(t\), for each candidate \((i,j)\), a predictor must output a probability \(p_{i,j,t}\), a conditional predicted arrival interval \(\hat h_{i,j,t}\), and a predicted RB demand \(\hat D_{i,j,t}\in\mathbb Z_{\ge0}\). **S interface, not a learned predictor formula:**

**(7)** \((p_{i,j,t},\hat h_{i,j,t},\hat D_{i,j,t})=\mathcal P(\mathcal H_{i,t},j),\quad 0\le p_{i,j,t}\le1,\quad p_{i,j,t}>0\Rightarrow \hat h_{i,j,t}\in\{t+1,\ldots,t+H\},\quad p_{i,j,t}=0\Rightarrow(\hat h_{i,j,t},\hat D_{i,j,t})=(t,0).\)

Here \(\mathcal H_{i,t}\) is the mobility/traffic history available strictly by decision time \(t\); \(\mathcal P\) is an externally implemented predictor. \(p_{i,j,t}\) must refer to a stated event horizon \(H\): “UE \(i\)'s **next** handover targets \(j\) during \((t,t+H]\).” For mutually exclusive next targets, **D**:

**(8)** \(\sum_{j\ne c_{i,t}}p_{i,j,t}\le1;\qquad p_{i,\varnothing,t}=1-\sum_{j\ne c_{i,t}}p_{i,j,t}.\)

\(\varnothing\) denotes no handover within the horizon. For a zero-probability target, the time/demand values in (7) are explicit numerical placeholders, not forecasts, and that target is ineligible for action selection. The notes request target/time/demand prediction but give no architecture or fitting equations for \(\mathcal P\). PreNS does provide a **slice demand** predictor, which may serve as a separate feature but is not a source for (7)'s mobility outputs.

For completeness, the relevant **P, Sec. IV-A, unnumbered** PreDP equations are transcribed with slice index \(s\), history position \(\ell\), sequence length \(N=5\), and normalized slice traffic \(n_s\):

**(9)** \(X_s(t)=[n_s(t-5\Delta_p),n_s(t-4\Delta_p),\ldots,n_s(t-\Delta_p)],\qquad\Delta_p=0.1\,{\rm s}.\)

**(10)** \(h_{s,\ell}=[\overrightarrow h_{s,\ell};\overleftarrow h_{s,\ell}],\quad H_s=[h_{s,1},\ldots,h_{s,N}].\)

**(11)** \(e_{s,\ell}=\operatorname{score}(h_{s,\ell}),\qquad a_{s,\ell}=\frac{\exp(e_{s,\ell})}{\sum_{\ell'=1}^{N}\exp(e_{s,\ell'})}.\)

**(12)** \(z_s(t)=\sum_{\ell=1}^{N}a_{s,\ell}h_{s,\ell},\qquad \hat R_s(t)=f(z_s(t)).\)

\(\overrightarrow h,\overleftarrow h\) are forward/backward LSTM hidden vectors; \(H_s\) is their sequence; \(e\) is an attention score, \(a\) its normalized weight, \(z_s\) the context vector, \(\hat R_s\) future slice demand, and \(f\) an unspecified output layer. PreNS writes \(\hat R_i(t)=f(z(t))\) while also describing forecasts \(\hat R_i(t+k)\); the output horizon is therefore ambiguous in the source. The paper does **not** specify the score function, LSTM gate equations, output activation, training loss, or map from normalized traffic to UE RB demand. Thus (9)–(12) are a source architecture sketch, not a fully specified predictor.

## 3. Uncertainty and confidence

PreNS **P, Sec. IV-B, unnumbered** gives the following formulas. Let \(\hat R_{s,\nu}\) and \(R_{s,\nu}\) be predicted and actual slice RB demand at historical interval \(\nu\), \(W\) the window length, \(\epsilon_R>0\) a denominator constant, and \(N_{s,\nu}\) a real-time metric vector:

**(13)** \(C^P_{s,t}=\left[\epsilon_R+\frac1W\sum_{\nu=t-W}^{t-1}|\hat R_{s,\nu}-R_{s,\nu}|\right]^{-1}.\)

**(14)** \(V^P_{s,t}=\frac1W\sum_{\nu=t-W}^{t-1}\operatorname{Var}(N_{s,\nu}).\)

**(15)** \(w^P_{s,t}=\frac{C^P_{s,t}}{C^P_{s,t}+V^P_{s,t}},\quad \bar w^P_{s,t}=1-w^P_{s,t},\quad G^P_{s,t}=w^P_{s,t}\hat R_{s,t}+\bar w^P_{s,t}N_{s,t}.\)

The superscript \(P\) means *paper*, not probability. Equation (13) is shown for a scalar slice and absolute error; the PDF says an element-wise norm such as \(L_1\). **Do not implement (14)–(15) literally:** the paper does not define the variance axis of a vector at one time, \(C^P\) has units RB\(^{-1}\), variance has squared metric units, and the two cannot be added. Further, high variance means *less* stability although (15) gives it *more* real-time weight. Finally, \(N\) contains metrics such as user count and QoS that cannot be added to RB demand. These are source fidelity findings, not silent corrections.

For the actual controller, keep uncertainty as *separate observed, dimensionless inputs* rather than falsely identifying inverse error with a calibrated handover probability. Given historical target labels \(y_{i,j,\nu}\in\{0,1\}\) and issued probabilities \(p_{i,j,\nu}\), a directly measurable **S** recent Brier error is

**(16)** \(u^{\rm target}_{i,j,t}=\frac1{|\mathcal W_{i,j,t}|}\sum_{\nu\in\mathcal W_{i,j,t}}(p_{i,j,\nu}-y_{i,j,\nu})^2,\qquad |\mathcal W_{i,j,t}|>0.\)

\(\mathcal W_{i,j,t}\) contains only predictions whose \(H\)-horizon outcomes are already known. \(u^{\rm target}\in[0,1]\) is error, not a confidence probability. When no matured observations exist, set the numerical uncertainty input to zero and its missingness flag to one; this zero is only a placeholder and must not be interpreted as perfect prediction. For demand, use **S** absolute prediction error normalized by target capacity:

**(17)** \(u^{\rm demand}_{i,j,t}=\frac1{|\mathcal W^D_{i,j,t}|}\sum_{\nu\in\mathcal W^D_{i,j,t}}\frac{|\hat D_{i,j,\nu}-D_{i,j,\nu}|}{K_j},\qquad K_j>0,\ |\mathcal W^D_{i,j,t}|>0.\)

\(D_{i,j,\nu}\) is realized demand in RBs for a completed handover; \(\mathcal W^D\) contains only such cases. This nonnegative quantity need not be at most one if demand can exceed cell capacity. No uncertainty estimate for arrival time is derivable without time-labeled data; its input is left as an optional measured feature, not an invented formula. The observed errors (16)–(17) do not change or calibrate \(p\) by themselves.

## 4. Handover and resource reservation

The notes' decision is **N, `text2.txt`**, normalized to one candidate and interval:

**(18)** \(a_{i,j,t}=(b_{i,j,t},q_{i,j,t},d_{i,j,t}),\quad b_{i,j,t}\in\{0,1\},\ q_{i,j,t}\in\mathbb Z_{\ge0},\ d_{i,j,t}\in\mathbb Z_{\ge0}.\)

\(b\) says whether to create a reservation, \(q\) is its RB amount, and \(d\) is activation delay in intervals. The activation interval is **D**:

**(19)** \(\tau^{\rm on}_{i,j,t}=t+d_{i,j,t}.\)

The source timing bound \(0\le T^{\rm reserve}_{i,j}\le T^{\rm HO}_{i,j}\) is ambiguous about absolute versus relative time. The implementable form is **S**, conditional on positive reservation:

**(20)** \(b_{i,j,t}=1\ \Longrightarrow\ p_{i,j,t}>0,\quad 0\le d_{i,j,t}\le\hat h_{i,j,t}-t,\quad 1\le q_{i,j,t}\le\hat D_{i,j,t}.\)

If \(b=0\), set \(q=d=0\). This is N's “reservation cannot exceed predicted demand” and “reserve before expected handover” under explicit interval units. It means an underestimated demand cannot be fully pre-reserved; this is a design constraint from the note, not a reliability guarantee. If \(\hat h<t\) or \(\hat D=0\), only \(b=0\) is legal.

The notes do not specify release/expiry or conflicting future reservations. To make capacity enforceable, use the following **S accounting contract**, which must be adopted explicitly by the simulator. At each interval start, activate due reservations after rechecking capacity; then process handovers; then release/convert affected reservations at the **end** of that interval. A reservation made at \(t\) is eligible to activate at \(\tau^{\rm on}\); if activation is rejected, it never holds RBs. Once activated, it remains held until the end of the first interval containing the UE's next handover, cancellation, or the expiry interval \(t+H\), inclusive. If handover occurs before activation, cancel the pending reservation without holding capacity. Write \(I_{i,j,t}^{\rm hold}(u)\in\{0,1\}\) for actual held status after activation and immediately before end-of-interval release. Then **D**:

**(21)** \(B_{i,j,u}=\sum_{t\le u}q_{i,j,t}I_{i,j,t}^{\rm hold}(u),\qquad A_{j,u}+\sum_iB_{i,j,u}\le K_j.\)

At most one pending or live reservation for an \((i,j)\) pair is allowed; otherwise the sum in (21) deliberately counts all contracts. Scheduled actions must pass the capacity check at *every* future candidate active interval using known commitments; future background allocation is unknown, so the runtime scheduler must revalidate before activation and may reject or cancel the contract if capacity is unavailable. The environment must serialize simultaneous activations in a fixed order and admit each only if (1) remains true. This cancellation/admission policy is an environment behavior, not a guarantee from prediction.

At a realized handover in interval \(h_i\), let \(j_i\) be the actual target, \(D_i\) its realized RB demand, \(B_i^{\rm live}=B_{i,j_i,h_i^-}\) the live reserved RBs immediately before conversion, and \(F_i^{\rm live}\) the uncommitted RBs there at that instant. **S accounting:**

**(22)** \(X_i=\min\{D_i,\ B_i^{\rm live}+F_i^{\rm live}\},\qquad U_i^{\rm used}=Y_i^{\rm HO}\min\{D_i,B_i^{\rm live}\},\qquad U_i^{\rm unused}=B_i^{\rm live}-U_i^{\rm used}.\)

\(X_i\) is potentially serviceable RBs, \(U_i^{\rm used}\) is used reservation, and \(U_i^{\rm unused}\) is unused reservation. Actual handover success cannot be defined as \(X_i=D_i\) if signaling/radio failure is possible; use the observed success flag \(Y_i^{\rm HO}\) from the environment. A failed handover uses zero reserved RBs even if capacity was held. Equation (22) excludes reservation at a *wrong* target from handover use, but it still consumes capacity until cancellation/expiry. A no-handover episode likewise uses zero reserved RBs.

The **N** objective and capacity/demand constraints, written with measurable terms, are:

**(23)** \(\max_{\pi}\ \alpha\,\mathbb E[\mathsf H]+\beta\,\mathbb E[\mathsf Q]-\gamma\,\mathbb E[\mathsf W]-\delta\,\mathbb E[\mathsf C],\qquad \alpha,\beta,\gamma,\delta\ge0.\)

\(\mathsf H\) is handover success, \(\mathsf Q\) QoS/SLA satisfaction, \(\mathsf W\) unused reservation cost, and \(\mathsf C\) reservation occupancy cost; \(\pi\) is the decision policy. The note supplies neither units, exact measurement windows, nor weights, so (23) is a goal template. Concrete dimensionless measurements appear in Section 6.

The note's \(R_{i,j,t}\ge R_i^{\rm SLA}\) cannot be treated as a hard constraint on **pre-reservation**: it conflicts with the permitted no-reserve action, uncertain targets, and insufficient capacity. It can apply to *actual post-handover allocation*, and only when feasible. For a resource floor \(D_i^{\rm SLA}\) in RBs, **S**:

**(24)** \(Y_i^{\rm HO}=1\ \land\ D_i^{\rm SLA}\le K_{j_i}-A_{j_i,h_i}^{\rm other}-B_{j_i,h_i}^{\rm other}\ \Longrightarrow\ A_{i,j_i,h_i}\ge D_i^{\rm SLA}.\)

\(A^{\rm other}\) and \(B^{\rm other}\) are allocation and live reservations for other UEs at the destination immediately before handover. If the antecedent is false, record SLA violation/infeasibility rather than declaring an impossible hard constraint. Throughput and latency SLA remain (4), measured after handover. Equation (24) is a proposed conditional admission rule, not a claim that the scheduler will enforce it automatically.

## 5. PPO policy and update

One PPO decision step addresses a single eligible \((i,j)\) in one physical interval \(t\). This specification permits **at most one new decision per interval**. If several candidates are eligible, process the one with the earliest predicted arrival \(\hat h_{i,j,t}\), breaking ties by stable UE and cell IDs; defer the others to later intervals and recompute their predictions then. If none is eligible, execute the no-op action. This proposed scheduling rule makes the PPO step index and physical interval index identical. It limits controller throughput and should be replaced by a separately specified joint or sequential multi-action policy if the deployment requires more decisions per interval. The state is **S**, based on `text2.txt` and available measurements:

**(25)** \(s_{i,j,t}=[p_{i,j,t},\hat h_{i,j,t}-t,\hat D_{i,j,t},F_{j,t},A_{j,t}/K_j,T_i^{\min},L_{s(i)}^{\max},u^{\rm target}_{i,j,t},u^{\rm demand}_{i,j,t},m^{\rm target},m^{\rm demand},s(i),\mathcal M_{i,j,t}].\)

\(m\)'s are binary missingness flags. \(\mathcal M_{i,j,t}\) is the deterministic admissibility mask derived from current and scheduled commitments; it can be represented as a variable-length action list. Categorical slice \(s(i)\) must be encoded for the policy. All continuous inputs are used in the units declared above; any learned normalization is an experiment option and must be fitted on training data only. The policy must see only data available at \(t\). For an interval with no eligible candidate, use a dedicated no-candidate state with only the no-op action admissible.

To avoid a continuous-action distribution and rounding ambiguity, **S** enumerates a finite set \(\mathcal A(s_{i,j,t})\) of admissible triples (18) satisfying (20) and the current/scheduled capacity checks. It always contains \((0,0,0)\). Action masking gives **S**:

**(26)** \(\pi_\theta(a\mid s)=\frac{\exp(\ell_\theta(s,a))}{\sum_{a'\in\mathcal A(s)}\exp(\ell_\theta(s,a'))}\quad(a\in\mathcal A(s));\qquad \pi_\theta(a\mid s)=0\quad(a\notin\mathcal A(s)).\)

\(\ell_\theta\) is the actor's logit, \(\theta\) its parameters. Save the exact admissibility mask and old masked log probability with every rollout transition; reuse that saved mask during PPO updates. Log probabilities refer to the *masked* distribution. Enumeration may be large. Candidate actions must be checked against all committed intervals from activation to expiry. A smaller fixed action catalog is possible only as a separately documented approximation.

Let \(r_t\) be the dimensionless reward of Section 6, \(\gamma_{\rm RL}\in[0,1]\) the RL discount, \(V_\psi(s)\) the critic, \(d_t\in\{0,1\}\) the terminal flag, \(\lambda\in[0,1]\) the GAE parameter, and \(T\) the rollout length. **G, finite-rollout form:**

**(27)** \(\delta_t=r_t+\gamma_{\rm RL}(1-d_t)V_\psi(s_{t+1})-V_\psi(s_t).\)

**(28)** \(\hat A_t=\sum_{l=0}^{T-t-1}(\gamma_{\rm RL}\lambda)^l\left[\prod_{k=0}^{l-1}(1-d_{t+k})\right]\delta_{t+l}.\)

The empty product is one. \(\hat A_t\) is the advantage estimate; it has reward units, here dimensionless. **D** value target:

**(29)** \(\hat V_t=V_\psi(s_t)+\hat A_t.\)

For the frozen rollout policy \(\theta_{\rm old}\), **O, eqs. (6)–(7):**

**(30)** \(r_t^{\pi}(\theta)=\exp[\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_{\rm old}}(a_t\mid s_t)].\)

**(31)** \(L^{\rm clip}(\theta)=\frac1T\sum_{t=0}^{T-1}\min\{r_t^{\pi}(\theta)\hat A_t,\ \operatorname{clip}(r_t^{\pi}(\theta),1-\epsilon_{\rm clip},1+\epsilon_{\rm clip})\hat A_t\}.\)

Here \(\epsilon_{\rm clip}\in(0,1)\) is PPO's dimensionless clip parameter, distinct from \(\epsilon_R\) in (13). The value loss is **O, PPO paper eq. (9), with a separate critic parameter \(\psi\)**:

**(32)** \(L^V(\psi)=\frac1T\sum_{t=0}^{T-1}[V_\psi(s_t)-\hat V_t]^2.\)

Compute (27)–(29) once from the critic snapshot used at rollout end; save and hold \(\hat A_t,\hat V_t\) fixed during the update. Optimize actor by maximizing (31) and critic by minimizing (32). The PPO paper also describes entropy bonuses and multiple epochs; their coefficients, epoch count, batch size, optimizer and network sizes are **not** supplied for this project and must be fixed in an experiment configuration. DDQN replay buffer, target network, and epsilon-greedy formulas in PreNS are **not PPO formulas** and are intentionally excluded.

## 6. Reward and outcome accounting

Because delayed handover outcomes occur after the reservation action, use a simulator that assigns reward to each interval and retains a contract ID to attribute the resolved outcome to its decision. Let \(\mathcal E_t\) be handovers resolved during interval \(t\), \(\mathcal V_t\) served-UE QoS evaluations in that interval, and \(\mathcal R_t^{\rm end}\) activated reservation contracts released or converted in that interval. Define \(h_t=|\mathcal E_t|^{-1}\sum_{i\in\mathcal E_t}Y_i^{\rm HO}\) if \(\mathcal E_t\ne\varnothing\), else \(h_t=0\). Define \(q_t=|\mathcal V_t|^{-1}\sum_{v\in\mathcal V_t}\mathbf1[T_v\ge T_v^{\min}\land L_v\le L_{s(v)}^{\max}]\) if \(\mathcal V_t\ne\varnothing\), else \(q_t=0\). Zero in an empty interval means *no positive reward event*, not failure. Let \(q_r^{\rm act}\) be the RB amount of a contract that actually activated, and zero for a contract that never activated. For a multi-cell reference capacity \(K^{\rm ref}=\sum_jK_j>0\), the **S** costs are:

**(33)** \(w_t=\frac{\sum_{r\in\mathcal R_t^{\rm end}}(q_r^{\rm act}-U_r^{\rm used})}{K^{\rm ref}},\qquad c_t=\frac{\sum_{i,j}B_{i,j,t}}{K^{\rm ref}}.\)

\(U_r^{\rm used}\) is its used RBs, so \(w_t\) charges unused *activated* RBs once, at resolution; \(c_t\) charges occupied RBs each interval. Both are dimensionless, but \(w_t\) can exceed one if several contracts end in the same interval; no clipping is implied. A rejected pending action has neither occupancy nor waste cost. Episode boundaries must settle or censor live reservations; no reward may use future outcomes before they occur.

**N, made dimensionally operational as S:**

**(34)** \(r_t=\alpha h_t+\beta q_t-\gamma w_t-\delta c_t,\qquad \alpha,\beta,\gamma,\delta\ge0.\)

These weights are dimensionless and must be recorded with the experiment. The zero convention for intervals without resolved handovers or QoS evaluations is defined above. The notes request penalties for failed handovers, SLA violation, and blocking other slices. Equation (34) **rewards success and QoS** and charges held/wasted capacity; failed events also forgo their corresponding positive rewards. A separate negative failure or blocking penalty is unsupported and must not be silently claimed. If the simulation measures blocked requests, it can be separately added only after its denominator and weight are specified.

For comparison, **P, Sec. III-B, eqs. (5)–(9)** defines normalized user weights \(\phi_i\ge0\) within slice \(s\), throughput utility \(G_i\), slice utility \(\Phi_s\), and total utility \(\Phi\):

**(35)** \(\sum_{i:s(i)=s}\phi_i=1,\qquad G_i=\phi_iT_i,\qquad\Phi_s=\sum_{i:s(i)=s}G_i,\qquad\Phi=\sum_s\Phi_s.\)

\(G_i\) and \(\Phi\) have throughput units; they cannot be added to the dimensionless terms of (34) without an explicit normalization. **P, Sec. IV-C** gives a DDQN reward \(\sum_s(\alpha_P U_s-\beta_P P_s)\) and fairness penalty \(P_s=\gamma_P\max(0,(\mu_s^{\min}-\mu_s)/\mu_s^{\min})\). Record these as:

**(36)** \(r_t^P=\sum_s(\alpha_P U_{s,t}-\beta_P P_{s,t}),\qquad P_{s,t}=\gamma_P\max\!\left(0,\frac{\mu_s^{\min}-\mu_{s,t}}{\mu_s^{\min}}\right),\ \mu_s^{\min}>0.\)

\(U_{s,t}\) is unspecified slice utility, \(P_{s,t}\) its penalty, \(\mu_{s,t}\) allocated slice share, and \(\alpha_P,\beta_P,\gamma_P\) paper weights. The paper says \(U\) reflects QoS, throughput and satisfaction but gives no formula. Equation (36) is therefore **not** a complete implementable reward. It is included for source coverage and should not replace (34).

**P, Sec. III-C, paper eqs. (9)–(10), in this document's allocation notation:**

**(37)** \(\max_{\{A_{j,s,t},A_{i,j,t}\}}\Phi\quad\text{subject to (4)–(6) and the weight conditions in (35)}.\)

This is PreNS's *slice-allocation* objective; it is not the handover-reservation objective (23). The paper states the objective for a single base station without a cell index; \(j\) here only makes the translation to the present notation explicit.

## 7. Evaluation metrics

Use disjoint train/validation/test trajectories and evaluate only **matured** predictions. Let \(\mathcal E\) be test handover events, \(N_{\rm HO}=|\mathcal E|\), \(\mathcal V\) served UE evaluations, and \(\mathcal R\) completed reservation contracts. Denominators must be nonzero; otherwise report “undefined,” not zero.

**P, Sec. VI-B, with consistent observation index \(k\):** for actual scalar demand \(y_k\), predicted \(\hat y_k\), \(n\) pairs, and \(\bar y=n^{-1}\sum_k y_k\),

**(38)** \({\rm RMSE}=\sqrt{\frac1n\sum_{k=1}^n(y_k-\hat y_k)^2},\qquad {\rm MAE}=\frac1n\sum_{k=1}^n|y_k-\hat y_k|.\)

**(39)** \(R^2=1-\frac{\sum_{k=1}^n(y_k-\hat y_k)^2}{\sum_{k=1}^n(y_k-\bar y)^2},\qquad \sum_k(y_k-\bar y)^2>0.\)

RMSE and MAE inherit the unit of \(y\); \(R^2\) is dimensionless. Report separately for RB demand and handover time (if predicted). R² is undefined for constant outcomes.

**S outcome ratios, from the notes' requested objectives:**

**(40)** \({\rm HSR}=\frac{\sum_{i\in\mathcal E}Y_i^{\rm HO}}{N_{\rm HO}},\qquad {\rm QoSRate}=\frac{\sum_{v\in\mathcal V}\mathbf1[T_v\ge T_v^{\min}\ \land\ L_v\le L_{s(v)}^{\max}]}{|\mathcal V|}.\)

\(Y_i^{\rm HO}\) is observed handover success. \(T_v,L_v\) are achieved throughput/latency at evaluation \(v\). Also report SLA infeasibility separately, since a failure caused by physically insufficient capacity is different from a controller mistake.

**S reservation diagnostics:** with activated reservation amount \(q_r^{\rm act}\), used RBs at handover \(U_r^{\rm used}\), and wrong-target/no-HO flag \(Z_r\), let \(\mathcal R\) contain completed **activated** contracts only. Report activation rejection separately if measured.

**(41)** \({\rm WrongTargetRate}=\frac{\sum_{r\in\mathcal R}Z_r}{|\mathcal R|}.\)

For a resource-waste ratio, compare RB amounts to RB amounts:

**(42)** \({\rm UnusedReservationFraction}=\frac{\sum_{r\in\mathcal R}(q_r^{\rm act}-U_r^{\rm used})}{\sum_{r\in\mathcal R}q_r^{\rm act}},\qquad 0\le U_r^{\rm used}\le q_r^{\rm act}.\)

\(Z_r=1\) when the actual next target differs or no handover occurs before expiry. The denominator in (42) counts only positive-size activated reservations; if there are none, the fraction is undefined.

**S** time-average occupied reservation share and total resource utilization are:

**(43)** \({\rm ReservationOccupancy}=\frac{\sum_{t\in\mathcal T}\sum_{i,j}B_{i,j,t}}{\sum_{t\in\mathcal T}\sum_jK_j},\qquad {\rm RBUtilization}=\frac{\sum_{t\in\mathcal T}\sum_j A_{j,t}}{\sum_{t\in\mathcal T}\sum_jK_j}.\)

\(\mathcal T\) is the test set of intervals. Reserved but unallocated RBs are excluded from RB utilization and appear in reservation occupancy. These two ratios need not sum to 1 because free RBs remain.

Prediction diagnostics must include target classification and calibration, not only scalar-error metrics (38)–(39): **S**

**(44)** \({\rm TargetBrier}=\frac1M\sum_{m=1}^M\sum_{j\in\mathcal C_m\cup\{\varnothing\}}(p_{m,j}-y_{m,j})^2,\qquad \sum_jy_{m,j}=1.\)

\(M\) is the number of matured prediction instances, \(\mathcal C_m\) their candidate targets, and \(y_{m,j}\) the one-hot realized next-target/no-handover label within horizon \(H\). \(\mathcal C_m\) must cover every possible next target in that horizon; if candidates are filtered, include a separate “other target” class in both prediction and label vectors. This is dimensionless. Report separately by slice, target load, and prediction uncertainty if sample sizes permit. Latency evaluation should report a predeclared summary (e.g. mean and 95th percentile) on an explicitly specified population; the supplied files provide no unique latency metric formula beyond per-UE latency constraints.

**S handover-specific reservation diagnostics**, directly matching the notes' whether/when questions:

**(45)** \({\rm ActiveReservationCoverage}=\frac{\sum_{i\in\mathcal E}\mathbf1[B_i^{\rm live}>0]}{N_{\rm HO}}.\)

For \(\mathcal E^{\rm active}=\{i\in\mathcal E:B_i^{\rm live}>0\}\), let \(\tau_i^{\rm on}\) be that matching reservation's actual activation interval. **S:**

**(46)** \({\rm MeanReservationLead}=\frac{\Delta}{|\mathcal E^{\rm active}|}\sum_{i\in\mathcal E^{\rm active}}(h_i-\tau_i^{\rm on}),\qquad |\mathcal E^{\rm active}|>0.\)

Lead has seconds as its unit and is zero for activation and handover in the same interval. The value is undefined if no matching reservation was active. Let \(\mathcal B\) be resource admission requests from other UEs/slices observed during test, with \(Z_b^{\rm block}=1\) if request \(b\) is rejected due to insufficient target capacity. Let \(\mathcal D\) be pending reservation activations due during test, with \(Z_d^{\rm reject}=1\) if activation \(d\) is rejected at runtime. **S:**

**(47)** \({\rm BlockRate}=\frac{\sum_{b\in\mathcal B}Z_b^{\rm block}}{|\mathcal B|},\qquad {\rm ActivationRejectRate}=\frac{\sum_{d\in\mathcal D}Z_d^{\rm reject}}{|\mathcal D|}.\)

Each rate is undefined for an empty denominator. BlockRate alone does not prove causation by reservation; compare it to a matched no-reservation baseline under the same trace. The sources do not specify this counterfactual simulator, so (47) is diagnostic rather than a reward term.

## 8. First and second checks

### Check 1: extraction, notation, algebra, dimensions

- **Coverage:** relevant PreNS model constraints and objective (paper eqs. 1–10), attention predictor, confidence/fusion, state/action/reward and prediction metrics are present. Paper DDQN optimization is excluded because the project explicitly chooses PPO; the paper gives no complete DDQN target equation anyway. `text2.txt`'s objective, capacity, demand, SLA, timing, state/action/reward concepts are mapped to (1), (20), (23)–(25), (34).
- **Notation:** source \(R\) meant capacity, allocation, predicted demand and RL reward in different sections. Here \(K,A,B,\hat D,r_t\) are distinct. Source \(T\) meant throughput, history window, and horizon; here \(T_i,W,H\) are distinct. \(\epsilon_R\) and \(\epsilon_{\rm clip}\) are distinct.
- **Algebra/dimensions:** (13) has inverse-RB units if its inputs are RBs; (14) has variance units; therefore (15) is invalid unless normalized/redefined. Source fusion adds heterogeneous units. Source timing bound needed a common time origin. Source pre-reservation SLA bound conflicts with no-reserve actions. Waste (42) compares RBs with RBs; occupancy (43) compares RB-intervals with RB-intervals. Ratios have nonzero-denominator guards.
- **Causality:** all uncertainty windows contain only matured outcomes; actor state contains current observations; retrospective waste is scored only after resolution.

### Check 2: independent implementation walk-through

1. **No handover:** \(p_{\varnothing}\) covers this case; action may be no-op; a held reservation expires, has \(U^{\rm used}=0\), and contributes occupancy/waste.
2. **Wrong target:** reservation remains unavailable to other UEs until cancellation/expiry and is not counted as used at the actual destination.
3. **Early/late handover:** only active reservations count at arrival; an early arrival sees no inactive promised capacity; a late arrival may see expired capacity. Both are observable outcomes, not silently corrected forecasts.
4. **Full target cell:** action set still contains no-op; impossible reservations are masked; no inequality requires more RBs than available.
5. **Several UEs:** (1) and (21) aggregate all held RBs. New decisions are serialized to one per interval; due activations from earlier decisions are serialized in stable UE/cell order and rechecked against (1).
6. **PPO rollout:** log probabilities from the same masked action set are saved; terminal flags stop GAE leakage; rollout policy is frozen during its update; future reward is not inserted into an earlier state.

Both checks **pass for the defined conditional reservation controller**. The predictor and environment contracts below remain prerequisites to a full executable system; no source-backed equation fills them.

### Extended numerical verification: 36 independent checks

The extended review ran from 2026-09-30 08:58:35 to 09:29:09 UTC (30 minutes 34 seconds). The following small-case checks were evaluated directly against the final formulas, including both signs of the PPO advantage. “Pass” means the stated identity/bound held for that case; these examples do not establish statistical performance or fill the missing predictor contract.

| # | Check | Expected and observed result |
|---:|---|---|
| 1 | Capacity equality | 6 allocated + 3 + 1 reserved = 10 RB; pass |
| 2 | Free capacity | 10 − 6 − 3 − 1 = 0 RB; pass |
| 3 | Capacity violation | 6 + 3 + 2 > 10 must be rejected; pass |
| 4 | Target probabilities | 0.4 + 0.35 leaves 0.25 no-HO; pass |
| 5 | Binary Brier perfect forecast | Score 0; pass |
| 6 | Multiclass Brier certain wrong forecast | Two-class score 2; pass |
| 7 | Demand-error normalization | 3 RB error / 10 RB = 0.3; pass |
| 8 | No-op action | Always in masked action set; pass |
| 9 | Zero predicted demand | No positive reservation can satisfy (20); pass |
| 10 | Failed handover resource use | Success flag 0 gives 0 used RBs; pass |
| 11 | Successful handover resource use | Demand 2, held 3 gives 2 used RBs; pass |
| 12 | Failed handover unused RBs | Held 3, used 0 gives 3 unused RBs; pass |
| 13 | Terminal TD residual | With reward 4 and value 2, residual = 2; pass |
| 14 | Two-step GAE | Residuals 2.8 and 2, discount 0.9, lambda 0.8 give 4.24; pass |
| 15 | Critic target | Value 1 + advantage 4.24 = 5.24; pass |
| 16 | PPO old-policy ratio | Equal old/new probabilities give 1; pass |
| 17 | Positive advantage, high ratio | A=2, ratio=1.5, clip 0.2 gives 2.4; pass |
| 18 | Positive advantage, low ratio | A=2, ratio=0.5 gives 1; pass |
| 19 | Negative advantage, low ratio | A=−2, ratio=0.5 gives −1.6; pass |
| 20 | Negative advantage, high ratio | A=−2, ratio=1.5 gives −3; pass |
| 21 | Wrong-target waste | Activated 3 at wrong cell, used 0 gives 3 RBs; pass |
| 22 | Never-activated contract waste | Activated 0, used 0 gives 0 RBs; pass |
| 23 | Unused fraction range | Contracts (3,2 used) and (4,0 used) give 5/7; pass |
| 24 | Occupancy bound | 4 reserved of 10 total is 0.4; pass |
| 25 | RMSE | Actual [1,2,3], forecast [1,3,2] gives √(2/3) RB; pass |
| 26 | MAE | Same data gives 2/3 RB; pass |
| 27 | R² | Same data gives 0; pass |
| 28 | Handover success rate | [1,0,1] gives 2/3; pass |
| 29 | QoS conjunction | One of two UEs meets both tests, giving 0.5; pass |
| 30 | Wrong-target rate | [1,0,1] gives 2/3; pass |
| 31 | Unused fraction upper bound | All 7 activated RBs unused gives 1; pass |
| 32 | RB utilization | 10 allocated over 20 RB-interval capacity = 0.5; pass |
| 33 | Reservation occupancy | 4 held over 20 RB-interval capacity = 0.2; pass |
| 34 | Active reservation coverage | Two of three handovers = 2/3; pass |
| 35 | Mean reservation lead | Leads 2 and 4 intervals, Δ=0.2 s gives 0.6 s; pass |
| 36 | Admission/rejection ratios | Two of five blocked; one of four activations rejected; 0.4 and 0.25; pass |

Additional source checks compared PreNS Sections III–IV and VI against (4)–(15), (35)–(39), the original PPO paper's eqs. (7), (9), (11)–(12) against (27)–(32), and the GAE paper's eq. (16) against (28). Revisions made during this extended review: corrected used-reservation accounting on failure, excluded never-activated contracts from waste, fixed event ordering, added zero-probability target eligibility, saved rollout masks/critic targets, repaired an action-count claim, made the source PreDP output transcription literal, and added handover-specific evaluation metrics. The source-model gaps in Section 10 remain open.

## 9. Complete symbol and notation table

| Symbol | Meaning; unit/domain |
|---|---|
| \(i,j,s,t,u,\nu,k,\ell,\ell',m\) | UE, cell, slice, decision interval, future interval, historical interval, horizon index, history position and dummy history position, evaluation sample index; indices |
| \(\mathcal U,\mathcal C,\mathcal S,\mathcal T,\mathcal E,\mathcal V,\mathcal R,\mathcal E^{\rm active},\mathcal B,\mathcal D\) | UE, cell, slice, test-interval, handover-event, served-evaluation, completed-contract, active-reservation handover, other admission request, due activation sets |
| \(c_{i,t},s(i),j_i,h_i\) | UE's source cell, slice, realized target, realized handover interval |
| \(\Delta,\Delta_p,H,W,N,T\) | scheduling interval length (s); prediction sample interval (s); handover horizon (intervals); error-window length (observations); PreDP input length (observations); PPO rollout length (steps) |
| \(K_j,A_{j,t},A_{j,s,t},A_{i,j,t},A^{\rm other},B^{\rm other},F_{j,t}\) | cell capacity, existing allocation, slice allocation, UE allocation, others' allocation/reservations, free capacity; RBs/interval |
| \(B_{i,j,t},B_{j,s,t},M_{j,s,t}\) | live UE reservation, slice reservation, enforceable slice floor; RBs/interval |
| \(T_{i,t},T_i^{\min},L_{i,t},L_s^{\max}\) | achieved/required throughput (bit/s), achieved/maximum latency (s) |
| \(\rho_{j,s,t},\rho_{i,j,t}\) | allocation fractions; dimensionless |
| \(\mathcal H_{i,t},\mathcal P,p_{i,j,t},p_{i,\varnothing,t}\) | available history, external predictor, target probability, no-HO probability |
| \(\hat h,\hat D,D,D_i^{\rm SLA}\) | predicted arrival interval, predicted demand, actual demand, RB SLA floor; demand quantities in RBs |
| \(X_s,n_s,h_{s,\ell},H_s,e_{s,\ell},a_{s,\ell},z_s,\operatorname{score},f,R_s,\hat R_s\) | PreDP history, normalized traffic, hidden state, hidden-state sequence, score, attention weight, context, scoring/output layers, actual/predicted slice demand |
| \(C^P,V^P,w^P,\bar w^P,G^P,N_{s,\nu},\epsilon_R\) | paper inverse-error, paper variance, paper fusion weights, paper fused value, observation vector, positive denominator constant |
| \(y_{i,j,\nu},\mathcal W,\mathcal W^D,u^{\rm target},u^{\rm demand},m^{\rm target},m^{\rm demand}\) | target label, matured error windows, measured errors, missingness flags |
| \(a_{i,j,t},b,q,d,\tau^{\rm on},I^{\rm hold}\) | action triple, reserve flag, amount (RBs), delay (intervals), activation, active flag |
| \(B_i^{\rm live},F_i^{\rm live},X_i,U_i^{\rm used},U_i^{\rm unused},Y_i^{\rm HO},Z_r\) | held/free destination RBs immediately before handover, serviceable RBs, used/unused reservation RBs, observed HO success, wrong-target/no-HO flag |
| \(s_{i,j,t},\mathcal M_{i,j,t},\mathcal A,\pi_\theta,\ell_\theta,\theta,\theta_{\rm old}\) | PPO state, admissibility mask, admissible actions, policy, logit, current/frozen actor parameters |
| \(V_\psi,\psi,r_t,d_t,\gamma_{\rm RL},\lambda,\delta_t,\hat A_t,\hat V_t\) | critic/value parameters, reward, terminal flag, discount, GAE parameter, TD residual, advantage, critic target |
| \(r_t^\pi,\epsilon_{\rm clip},L^{\rm clip},L^V\) | policy likelihood ratio, clip width, actor objective, critic loss |
| \(\alpha,\beta,\gamma,\delta,h_t,q_t,w_t,c_t,K^{\rm ref},\mathcal E_t,\mathcal V_t,\mathcal R_t^{\rm end}\) | project reward weights and dimensionless components, reference capacity, resolved handovers, QoS evaluations, ending contracts in interval \(t\) |
| \(\phi_i,G_i,\Phi_s,\Phi,r_t^P,U_s,P_s,\mu_s,\mu_s^{\min},\alpha_P,\beta_P,\gamma_P\) | paper throughput weight/utility and paper DDQN reward/fairness symbols |
| \(y_k,\hat y_k,\bar y,n,M,N_{\rm HO}\) | actual/predicted scalar, sample mean, scalar/test prediction/handover sample counts |
| \(q_r,q_r^{\rm act},U_r^{\rm used}\) | requested, actually activated, and used RBs for reservation \(r\) |
| \(\mathsf H,\mathsf Q,\mathsf W,\mathsf C,\pi\) in (23) | note's aggregate handover success, QoS, waste, occupancy criteria and decision policy; (23) is source-only |
| \(\mathcal C_m,p_{m,j},y_{m,j},Z_r\) | exhaustive next-target classes for prediction case \(m\), including an aggregate other-target class if needed; forecast probability, one-hot outcome, wrong-target/no-HO contract flag |
| \(\tau_i^{\rm on},Z_b^{\rm block},Z_d^{\rm reject}\) | actual reservation activation interval, other request blocked flag, due activation rejection flag |

## 10. Assumptions and unspecified inputs

1. Cell RB capacities are expressed over the **same scheduling interval**; a reservation is admission control for that interval's RB budget, not a guarantee of a fixed frequency-time resource across all slots.
2. One UE belongs to one slice during the decision episode. The next-target labels are mutually exclusive; no-handover is an explicit class. A new decision after a handover starts a new episode/context.
3. Handover success, achieved throughput, achieved delay, allocations, and actual RB demand are measured by the testbed/simulator. The sources provide no channel, scheduler, traffic-to-RB, or handover-failure equations.
4. The environment must define cancellation, expiry, admission ordering, and handling of future commitments as described in Section 4. These are proposed accounting semantics, not claims about 3GPP or PreNS.
5. Required external predictor contract: a calibrated \(p\) for a fixed horizon; a conditional arrival interval; RB demand; their training data and fitting procedure. The source paper only specifies a partial **slice-level** Bi-LSTM sketch. This is the main unresolved requirement for an end-to-end implementation from raw mobility measurements.
6. Required experiment configuration: \(\Delta,H,W,K^{\rm ref}\), capacity/floor values, all reward weights, PPO discount/clip/GAE parameters, actor/critic architecture, optimizer/training schedule, state normalization, and missing-data convention. These are hyperparameters or environment inputs, not derivable constants.
7. No hard guarantee of SLA can be inferred from reservation counts alone; throughput, latency, signaling success, and competing traffic remain environment dependent.

## 11. Equation dependency and order

1. Configure sets, capacity, slice floors, intervals: (1)–(6).
2. Produce prediction inputs and optional PreDP slice forecast: (7)–(12).
3. Mature labels and compute measured uncertainty: (16)–(17). Treat (13)–(15) as source-only and invalid for direct fusion.
4. Build admissible reservation candidates and occupancy ledger: (18)–(21), using (1)–(2).
5. Build state and masked policy: (25)–(26), using (7), (16)–(17), (2), (20)–(21).
6. Execute action, observe handover and QoS: (22), (24), then reward (33)–(34).
7. Collect rollout and update PPO: (27)–(32), using (34).
8. Evaluate held-out data: (38)–(47).

## 12. Final consolidated list in implementation order

**Implement:** (1) capacity; (2) free RBs; (3) slice floors with declared semantics; (4) throughput/latency tests; (7) predictor contract; (8) next-target normalization; (16) target error; (17) demand error; (18) action; (19) activation; (20) legal action bounds; (21) live reservation ledger; (25) state; (26) masked PPO policy; (22) handover accounting; (24) feasible SLA floor; (33) occupancy/waste components; (34) reward; (27) TD residual; (28) GAE; (29) value target; (30) likelihood ratio; (31) clipped actor objective; (32) critic loss; (38)–(47) evaluation metrics.

**Source reference only:** (5)–(6) PreNS allocation ratios; (9)–(12) incomplete PreDP architecture; (13)–(15) dimensionally invalid PreNS confidence/fusion as printed; (23) uninstantiated note objective; (35) PreNS throughput utility; (36) incomplete PreNS DDQN reward; (37) PreNS constrained slice-allocation objective.

The controller equations are internally checkable and implementable **once** the declared external predictor, simulator, and hyperparameters are supplied. Those components cannot be reconstructed uniquely from the provided files without unsupported assumptions.
