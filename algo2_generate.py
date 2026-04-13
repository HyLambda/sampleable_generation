"""
algo2_generate.py

certified barycentric generation (algorithm 2 of [lam2026a]).

given a trained cloud Y = T_theta(X) with L_beta^{(k)}(Y) close to 1
and bracket [m_k, M_k] = [min_i S_k(i;Y), max_i S_k(i;Y)],
this generates new certified samples.

one generation attempt:

  1. draw i* ~ Uniform{1,...,N}
     (unbiased because L_beta^{(k)}(Y) ~ 1)

  2. retrieve N_k(i*; Y), the k nearest neighbors of y_{i*} in Y

  3. form the barycentric proposal
         y_hat = (y_{i*} + sum_{j in N_k(i*)} y_j) / (k+1)

     equivalently (proposition 4.3 of [lam2026a]):
         y_hat = c + k/(k+1) * v_bar
     where
         c     = y_{i*}
         v_bar = (1/k) sum_{j in N_k(i*)} (y_j - y_{i*})

  4. query S_k(y_hat; Y) using the stored cloud Y

  5. accept iff m_k <= S_k(y_hat; Y) <= M_k
     on acceptance: x_hat = T_theta^{-1}(y_hat)

geometric identities used in the acceptance filter
(proposition 4.3 and corollary 4.7 of [lam2026a]):

    ||y_hat - c||^2 = k^2/(k+1)^2 * ||v_bar||^2

    (1/k) sum_{j in B*} ||y_hat - y_j||^2
         = beta - k(k+2)/(k+1)^2 * ||v_bar||^2

    where beta = S_k(i*; Y)  and  B* = N_k(i*; Y)

    bracket:  beta/(k+1)^2  <=  (1/k) sum  <=  beta

hy p. g. lam
worcester polytechnic institute
hlam@wpi.edu
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
from sklearn.neighbors import NearestNeighbors


Array = np.ndarray


# ---------------------------------------------------------------------------
# bracket precomputation
# ---------------------------------------------------------------------------

def compute_bracket(Y: Array, k: int) -> tuple[Array, Array, float, float]:
    """
    precompute the bracket [m_k, M_k] and all S_k(i; Y).

    returns (nbr_idx, sk, mk, Mk) where
      nbr_idx[i] = (k,) indices of k nearest neighbors of y_i
      sk[i]      = S_k(i; Y) = (1/k) sum_{j in N_k(i)} ||y_i - y_j||^2
      mk         = min_i sk[i]
      Mk         = max_i sk[i]
    """
    N = len(Y)
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(Y)
    dist, idx = nn.kneighbors(Y)
    # strip self (column 0)
    idx  = idx[:, 1:]          # (N, k)
    dist = dist[:, 1:] ** 2    # (N, k) squared euclidean

    sk  = dist.mean(axis=1)    # (N,)
    mk  = float(sk.min())
    Mk  = float(sk.max())

    return idx, sk, mk, Mk


# ---------------------------------------------------------------------------
# S_k for an external point (proposal)
# ---------------------------------------------------------------------------

def sk_external(p: Array, Y: Array, k: int,
                nn_fitted: NearestNeighbors) -> float:
    """
    S_k(p; Y) = (1/k) sum_{j in N_k(p;Y)} ||p - y_j||^2
    for a point p not in Y.
    """
    dist, _ = nn_fitted.kneighbors(p[None, :], n_neighbors=k)
    return float((dist[0]**2).mean())


# ---------------------------------------------------------------------------
# barycentric proposal
# ---------------------------------------------------------------------------

def barycentric_proposal(i_star: int, Y: Array,
                          nbr_idx: Array) -> tuple[Array, Array, float]:
    """
    y_hat = (y_{i*} + sum_{j in N_k(i*)} y_j) / (k+1)

    also returns v_bar and ||v_bar||^2 for diagnostics.
    """
    k    = nbr_idx.shape[1]
    nbrs = Y[nbr_idx[i_star]]             # (k, d)
    v_bar = (nbrs - Y[i_star]).mean(0)    # mean neighbor displacement
    y_hat = Y[i_star] + k/(k+1) * v_bar
    return y_hat, v_bar, float(np.dot(v_bar, v_bar))


# ---------------------------------------------------------------------------
# single generation attempt
# ---------------------------------------------------------------------------

def one_attempt(Y: Array, k: int,
                nbr_idx: Array, sk: Array,
                mk: float, Mk: float,
                nn_fitted: NearestNeighbors,
                rng: np.random.Generator) -> dict:
    """
    one draw-propose-test cycle.

    returns dict with keys:
      i_star   -- anchor index drawn
      y_hat    -- barycentric proposal
      sq       -- S_k(y_hat; Y)
      accepted -- bool
      beta     -- S_k(i*; Y)
      vbar_sq  -- ||v_bar||^2
    """
    N = len(Y)
    i_star = int(rng.integers(0, N))

    y_hat, v_bar, vbar_sq = barycentric_proposal(i_star, Y, nbr_idx)

    beta = float(sk[i_star])
    sq   = sk_external(y_hat, Y, k, nn_fitted)

    accepted = (mk <= sq <= Mk)

    return dict(i_star=i_star, y_hat=y_hat, sq=sq,
                accepted=accepted, beta=beta, vbar_sq=vbar_sq)


# ---------------------------------------------------------------------------
# generate n_samples accepted points
# ---------------------------------------------------------------------------

def generate(Y: Array, k: int,
             n_samples: int,
             T_inv: Optional[Callable[[Array], Array]] = None,
             max_trials: Optional[int] = None,
             seed: Optional[int] = 0) -> dict:
    """
    run algorithm 2 until n_samples accepted proposals are collected.

    Y       : (N, d) trained cloud.
    k       : neighborhood size (should match the value used in training).
    n_samples: number of accepted samples to collect.
    T_inv   : optional inverse map T_theta^{-1}: R^d -> R^d.
              if None, returns proposals in Y-space only.
    max_trials: stop early after this many total attempts even if
               n_samples not reached. default: 50 * n_samples.

    returns dict with keys:
      samples    -- (n_samples, d) accepted Y-space samples
      x_samples  -- (n_samples, d) pullbacks via T_inv (or None)
      trials     -- total number of attempts
      accepted   -- total accepted
      accept_rate-- accepted / trials
      diagnostics-- per-accepted dict list with i_star, sq, beta, vbar_sq
    """
    if max_trials is None:
        max_trials = 50 * n_samples

    rng = np.random.default_rng(seed)

    nbr_idx, sk, mk, Mk = compute_bracket(Y, k)
    nn_fitted = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(Y)

    samples     = []
    diagnostics = []
    trials      = 0

    while len(samples) < n_samples and trials < max_trials:
        result = one_attempt(Y, k, nbr_idx, sk, mk, Mk, nn_fitted, rng)
        trials += 1
        if result['accepted']:
            samples.append(result['y_hat'])
            diagnostics.append({
                'i_star' : result['i_star'],
                'sq'     : result['sq'],
                'beta'   : result['beta'],
                'vbar_sq': result['vbar_sq'],
            })

    samples_arr = np.array(samples) if samples else np.empty((0, Y.shape[1]))

    x_samples = None
    if T_inv is not None and len(samples_arr) > 0:
        x_samples = np.array([T_inv(p) for p in samples_arr])

    return dict(
        samples     = samples_arr,
        x_samples   = x_samples,
        mk          = mk,
        Mk          = Mk,
        trials      = trials,
        accepted    = len(samples),
        accept_rate = len(samples) / max(trials, 1),
        diagnostics = diagnostics,
    )


# ---------------------------------------------------------------------------
# geometric identity check (proposition 4.3 of [lam2026a])
# ---------------------------------------------------------------------------

def check_identities(Y: Array, k: int, i_star: int,
                      nbr_idx: Array, sk: Array) -> dict:
    """
    verify the three geometric identities of proposition 4.3 numerically.

    returns a dict with lhs, rhs and absolute error for each identity.
    """
    y_hat, v_bar, vbar_sq = barycentric_proposal(i_star, Y, nbr_idx)
    beta  = sk[i_star]
    c     = Y[i_star]
    B_idx = nbr_idx[i_star]

    # identity 1: ||y_hat - c||^2 = k^2/(k+1)^2 * ||v_bar||^2
    lhs1 = float(np.dot(y_hat - c, y_hat - c))
    rhs1 = float((k**2 / (k+1)**2) * vbar_sq)

    # identity 2: (1/k) sum_{j in B*} ||y_hat - y_j||^2
    #            = beta - k(k+2)/(k+1)^2 * ||v_bar||^2
    avg_sq = float(((y_hat - Y[B_idx])**2).sum(axis=1).mean())
    lhs2   = avg_sq
    rhs2   = float(beta - k*(k+2)/(k+1)**2 * vbar_sq)

    # bracket bounds (corollary 4.7):
    #   beta/(k+1)^2  <=  avg_sq  <=  beta
    lo_bound = beta / (k+1)**2
    hi_bound = beta

    return dict(
        identity1 = dict(lhs=lhs1, rhs=rhs1, err=abs(lhs1-rhs1)),
        identity2 = dict(lhs=lhs2, rhs=rhs2, err=abs(lhs2-rhs2)),
        bracket   = dict(lo=lo_bound, val=avg_sq, hi=hi_bound,
                         in_bracket=(lo_bound <= avg_sq <= hi_bound)),
        beta      = float(beta),
        vbar_sq   = float(vbar_sq),
    )


# ---------------------------------------------------------------------------
# example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from algo1_train import train, Config

    rng = np.random.default_rng(42)
    N = 300

    n1 = round(N * 0.55)
    X = np.vstack([
        rng.normal([0.28, 0.52], 0.09, (n1, 2)),
        rng.normal([0.70, 0.42], 0.14, (N - n1, 2)),
    ])
    X = np.clip(X, 0.02, 0.98)

    # train T_theta (algo 1)
    cfg = Config(k=5, lam=0.3, tau=8, eta=0.005, n_iter=150, seed=0)
    train_result = train(X, cfg)
    Y = train_result['Y']

    print(f"training done.  L_beta final = {train_result['history'][-1]['lb']:.4f}")
    print(f"bracket: mk={train_result['mk']:.5f}  Mk={train_result['Mk']:.5f}")

    # generate 200 samples (algo 2)
    gen = generate(Y, k=cfg.k, n_samples=200, seed=1)

    print(f"\ntrials: {gen['trials']}   accepted: {gen['accepted']}")
    print(f"accept rate: {gen['accept_rate']:.3f}")

    # verify identities on one anchor
    nbr_idx, sk, mk, Mk = compute_bracket(Y, cfg.k)
    chk = check_identities(Y, cfg.k, i_star=0, nbr_idx=nbr_idx, sk=sk)
    print(f"\nidentity 1 error: {chk['identity1']['err']:.2e}")
    print(f"identity 2 error: {chk['identity2']['err']:.2e}")
    print(f"bracket satisfied: {chk['bracket']['in_bracket']}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(X[:, 0], X[:, 1], s=10, c='grey',      label='X (raw)')
    axes[0].scatter(Y[:, 0], Y[:, 1], s=10, c='steelblue', label='Y = T(X)')
    axes[0].set_title('trained cloud')
    axes[0].legend(fontsize=8)

    axes[1].scatter(Y[:, 0], Y[:, 1], s=10, c='steelblue', alpha=0.4, label='Y')
    S = gen['samples']
    axes[1].scatter(S[:, 0], S[:, 1], s=18, marker='D',
                    c='seagreen', label=f'{gen["accepted"]} accepted samples')
    axes[1].set_title('algorithm 2 output')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('algo2_result.png', dpi=150)
    plt.show()
