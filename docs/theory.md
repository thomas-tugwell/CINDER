# Theory

This document defines the objects the software manipulates and the equations each module
implements. Nothing here asserts that the approach is novel or that TMol scores are physical
free energies; it is a framework for testing the hypothesis in §14–16.

## 1. Problem definition

Given a fixed protein target, find binder sequences whose *ensemble* of complex conformations
scores well under a differentiable molecular energy, rather than scoring one minimised structure.

## 2. Variables

| Symbol | Meaning |
|---|---|
| $S$ | binder sequence (designable at a subset of positions) |
| $T$ | fixed target (sequence and structure) |
| $X$ | complex conformation (coordinates, or another fixed-size representation) |
| $E$ | energy of a conformation, $E = U_{\mathrm{TMol}}(X,S,T)$ |

## 3. Why $E$ is deterministic given $X, S, T$

TMol evaluates a fixed function of coordinates and chemistry. Once $(X,S,T)$ is known there is
no remaining randomness in $E$; all stochasticity lives in $X$ (and in $S$ at the outer level).

## 4. Joint distribution

$$
p(X,E\mid S,T) = p(X\mid S,T)\,\delta\!\left[E-U_{\mathrm{TMol}}(X,S,T)\right].
$$

The Dirac delta says $E$ is not independently sampled: it is computed from $X,S,T$.

## 5. Base structural distribution

$p_{\mathrm{base}}(X\mid S,T)$ is whatever generative or sampling model supplies conformations
(Langevin, flow, diffusion, flow matching, a pretrained model such as BioEmu2). It is accessed
only through the `ConformationModel` interface.

## 6. Energy-guided distribution

$$
p_{\mathrm{guided}}(X\mid S,T)\;\propto\;p_{\mathrm{base}}(X\mid S,T)\,\exp\!\left[-\lambda\,U_{\mathrm{TMol}}(X,S,T)\right].
$$

With no learned base (energy-only baseline) this is $p_U \propto \exp[-\lambda U]$.
$\lambda$ is a configurable **guidance coefficient**, not a calibrated inverse temperature.

## 7. Normalisation

$$
Z(S,T)=\int p_{\mathrm{base}}(X\mid S,T)\,e^{-\lambda U}\,dX ,\qquad
\log p_{\mathrm{guided}}=\log p_{\mathrm{base}}-\lambda U-\log Z .
$$

$Z$ is intractable in general and is never needed for sampling or for score-based guidance.

## 8. Gradient of the log guided density

$$
\nabla_X\log p_{\mathrm{guided}}=\nabla_X\log p_{\mathrm{base}}-\lambda\nabla_X U_{\mathrm{TMol}} .
$$

($\log Z$ is constant in $X$.) This is the bridge between learned score models and TMol
coordinate gradients; it is implemented once in `conformations/guidance.py`.

## 9. Langevin interpretation

The overdamped Langevin SDE $dX=\nabla_X\log p\,dt+\sqrt2\,dW$ has stationary law $p$. With
$\nabla\log p=-\lambda\nabla U$ the Euler–Maruyama step is

$$
X_{t+1}=X_t-\eta\lambda\nabla_X U(X_t)+\sqrt{2\eta}\,\epsilon_t,\qquad\epsilon_t\sim\mathcal N(0,I).
$$

The unadjusted scheme has $O(\eta)$ bias; the test-suite checks moments against closed forms
(harmonic, double well, Gaussian mixtures, and the product-of-Gaussians guided case).

## 10. Conditional normalising flows

A flow $q_\theta(X\mid S,T)$ has exact density. Training on an energy landscape minimises

$$
L(\theta)=\mathbb E_{X\sim q_\theta}\!\big[\lambda U(X)+\log q_\theta(X)\big]
=\mathrm{KL}(q_\theta\,\|\,p_U)-\log Z_U\;\ge\;-\log Z_U ,
$$

with equality iff $q_\theta=p_U$. This is a reverse KL, hence mode-seeking.

## 11. Diffusion

Forward process $X_t=\alpha_tX_0+\sigma_t\epsilon$; learned score $s_\theta(X_t,t,S,T)$;
guided score $s_{\mathrm{guided}}=s_\theta-\lambda\nabla_X U$. Ordinary diffusion lacks a
tractable normalised density, so `log_prob` is unsupported. *(Stub.)*

## 12. Flow matching

$\dfrac{dX}{dt}=v_\theta(X,t,S,T)$, heuristically guided by
$v_{\mathrm{guided}}=v_\theta-\lambda(t)\nabla_XU$. The exact guided velocity for an arbitrary
learned field is not this simple; treat it as a heuristic to be validated. *(Stub.)*

## 13. Outer sequence distribution

$q_\phi(S\mid T)=\prod_i q_\phi(S_i\mid T)$ with trainable logits. REINFORCE:

$$
\nabla_\phi J=\mathbb E\big[(R(S)-b)\,\nabla_\phi\log q_\phi(S\mid T)\big],\qquad
L_{\mathrm{policy}}=-(R-b)\log q_\phi(S)-\alpha H[q_\phi],
$$

with a batch-mean baseline $b$ and entropy weight $\alpha$ to prevent premature collapse.

## 14. Ensemble-level sequence objective

Sample $X_1,\dots,X_K\sim p(X\mid S,T)$ and aggregate the design energy:

$$
R(S)=-\frac1K\sum_{k=1}^K U_{\mathrm{design}}(X_k,S,T).
$$

Alternative aggregators: median, softmin, quantile, and the free-energy surrogate
$-\lambda^{-1}\log\frac1K\sum_k e^{-\lambda U_k}$. For tractable-density models the variational
free energy $F_q(S)=\mathbb E_{X\sim q_\theta}[U+\lambda^{-1}\log q_\theta]$ trades energy against
entropy.

## 15. Why total complex energy is insufficient

Total energy is dominated by intra-chain terms that do not depend on binding and by covalent
strain, and it rewards sequences that stabilise the *scaffold*. We therefore optimise
a composite

$$
U_{\mathrm{design}}=w_{\mathrm{int}}U_{\mathrm{interface}}+w_{\mathrm{rep}}U_{\mathrm{rep}}+w_{\mathrm{strain}}U_{\mathrm{strain}}+w_{\mathrm{restr}}U_{\mathrm{restraint}}+\cdots
$$

with each term logged separately. It is a *design surrogate*, not $\Delta G_{\mathrm{bind}}$.

## 16. Limitations of treating TMol as thermodynamic

TMol is a reimplementation of a Rosetta-style empirical score; its units are not calibrated to
$k_BT$, it omits conformational entropy and explicit solvent dynamics, and its terms are
partly knowledge-based. $e^{-\lambda U}$ therefore defines a *tilted ensemble*, not a Boltzmann
ensemble, unless calibration against experiment justifies more. Cartesian Langevin on all atoms
additionally heats stiff covalent terms (equipartition); see `docs/models.md`.
