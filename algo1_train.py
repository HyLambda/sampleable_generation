"""
algo1_train.py

training T_theta on the joint loss

    J(theta) = L_alpha(X, Y) + lambda * log L_tilde_{beta,tau}^{(k)}(Y)

where Y = T_theta(X).

notation follows [lam2026a] exactly.

L_alpha measures pairwise distance distortion:

    L_alpha(X, Y) = (1/N^2) sum_{i,j} | ||x_i - x_j||^2 - ||y_i - y_j||^2 |

L_beta^{(k)} measures local density nonuniformity:

    S_k(i; Y) = (1/k) sum_{j in N_k(i;Y)} ||y_i - y_j||^2
    L_beta^{(k)}(Y) = max_i S_k(i;Y) / min_i S_k(i;Y)

the smooth surrogate used in the gradient is (definition 4.1 of [lam2026a]):

    log L_tilde_{beta,tau}^{(k)}(Y)
        = (1/tau) * logsumexp(tau * log S_k(i;Y))
        + (1/tau) * logsumexp(-tau * log S_k(i;Y))
        - (2/tau) * log N

gradients (proposition 4.1 of [lam2026a]):

    d L_alpha / d y_k
        = -(4/N^2) sum_j sign(||x_k-x_j||^2 - ||y_k-y_j||^2) * (y_k - y_j)

    d log L_tilde / d y_k
        = (2/k) * [(w_k^+ - w_k^-) / S_k(k;Y)] * sum_{j in N_k(k)} (y_k - y_j)
        + (2/k) * sum_{i: k in N_k(i)} [(w_i^+ - w_i^-) / S_k(i;Y)] * (y_k - y_i)

    w_i^+ = softmax weights of  tau * log S_k(i;Y)
    w_i^- = softmax weights of -tau * log S_k(i;Y)

defect lower bound (corollary 3.3 of [lam2026a]):

    L_alpha >= k * m_k(X) / ((1 + R*) * N^2) * (R_0 - R*)

where R_0 = L_beta^{(k)}(X) and R* is any target ratio >= 1.

hy p. g. lam
worcester polytechnic institute
hlam@wpi.edu
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.neighbors import NearestNeighbors


Array = np.ndarray


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    k:         int   = 5       # neighborhood size
    lam:       float = 0.3     # weight lambda on log L_tilde
    tau:       int   = 8       # softmax sharpness in surrogate
    eta:       float = 0.005   # gradient step size
    grad_clip: float = 0.08    # per-point gradient norm clip
    n_iter:    int   = 200     # number of gradient steps
    # for large N: subsample at most this many pairs for L_alpha gradient
    max_grad_pairs: int = 20_000
    seed: Optional[int] = 0


# ---------------------------------------------------------------------------
# squared euclidean distances (vectorized)
# ---------------------------------------------------------------------------

def sq_dist_matrix(A: Array) -> Array:
    """
    returns D[i,j] = ||a_i - a_j||^2.
    shape: (N, N).
    uses the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a^T b.
    """
    norms = np.sum(A**2, axis=1, keepdims=True)   # (N,1)
    return norms + norms.T - 2.0 * A @ A.T


# ---------------------------------------------------------------------------
# k-NN and S_k
# ---------------------------------------------------------------------------

def build_knn(Y: Array, k: int) -> tuple[Array, Array]:
    """
    returns (indices, S_k values).
    indices[i] = array of k neighbor indices of y_i, excluding i itself.
    sk[i] = (1/k) sum_{j in N_k(i)} ||y_i - y_j||^2.
    """
    N = len(Y)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(Y)
    dist_sq, idx = nbrs.kneighbors(Y)
    # column 0 is self (distance 0); strip it
    idx    = idx[:, 1:]      # (N, k)
    dist_sq = dist_sq[:, 1:]  # (N, k), already squared? no, kneighbors returns euclidean
    # sklearn returns euclidean distance, not squared; square it
    dist_sq = dist_sq**2
    sk = dist_sq.mean(axis=1)   # (N,)
    return idx, sk


# ---------------------------------------------------------------------------
# L_alpha
# ---------------------------------------------------------------------------

def L_alpha(DX: Array, DY: Array) -> float:
    """
    L_alpha(X, Y) = (1/N^2) sum_{i,j} |DX[i,j] - DY[i,j]|
    where DX, DY are squared distance matrices.
    """
    N = DX.shape[0]
    return float(np.abs(DX - DY).sum()) / (N * N)


# ---------------------------------------------------------------------------
# L_beta^{(k)}
# ---------------------------------------------------------------------------

def L_beta(sk: Array) -> float:
    """
    L_beta^{(k)} = max_i S_k(i) / min_i S_k(i).
    equals 1 when the cloud is perfectly uniform.
    """
    return float(sk.max() / sk.min())


# ---------------------------------------------------------------------------
# log L_tilde surrogate and its gradient weights
# ---------------------------------------------------------------------------

def log_L_tilde(sk: Array, tau: int) -> tuple[float, Array, Array]:
    """
    log L_tilde_{beta,tau}^{(k)}(Y) (definition 4.1 of [lam2026a]).

    returns (lt, w_plus, w_minus) where
      lt     = value of the surrogate
      w_plus = softmax weights of  tau * log sk_i
      w_minus= softmax weights of -tau * log sk_i

    numerically stable via logsumexp shift.
    """
    N = len(sk)
    log_sk = np.log(np.maximum(sk, 1e-12))

    a_plus  =  tau * log_sk
    a_minus = -tau * log_sk

    # logsumexp
    mp = a_plus.max();   sp = np.exp(a_plus  - mp).sum()
    mn = a_minus.max();  sn = np.exp(a_minus - mn).sum()

    lt = (1.0/tau) * (mp + np.log(sp) - np.log(N)) \
       + (1.0/tau) * (mn + np.log(sn) - np.log(N))

    w_plus  = np.exp(a_plus  - mp) / sp
    w_minus = np.exp(a_minus - mn) / sn

    return float(lt), w_plus, w_minus


# ---------------------------------------------------------------------------
# gradient of log L_tilde w.r.t. Y (proposition 4.1 of [lam2026a])
# ---------------------------------------------------------------------------

def grad_log_L_tilde(Y: Array, idx: Array, sk: Array,
                      w_plus: Array, w_minus: Array, k: int) -> Array:
    """
    d log L_tilde / d y_i for all i.
    shape: (N, d).

    two terms per point i (proposition 4.1):
      term 1: i owns the neighborhood -- contribution from d S_k(i;Y) / d y_i
      term 2: i appears inside a neighbor set -- contribution via reverse neighbors
    """
    N, d = Y.shape
    coeff = (w_plus - w_minus) / np.maximum(sk, 1e-12)   # (N,)
    c2 = 2.0 / k

    G = np.zeros((N, d))

    # term 1: anchor i, neighbors j in N_k(i)
    for i in range(N):
        t1 = coeff[i] * c2
        diffs = Y[i] - Y[idx[i]]         # (k, d)
        G[i] += t1 * diffs.sum(axis=0)

    # term 2: reverse: for each i, find which points have i in their N_k
    # build reverse neighbor list
    rev = [[] for _ in range(N)]
    for i in range(N):
        for j in idx[i]:
            rev[j].append(i)

    for j in range(N):
        for i in rev[j]:
            t2 = coeff[i] * c2
            G[j] += t2 * (Y[j] - Y[i])

    return G


# ---------------------------------------------------------------------------
# gradient of L_alpha w.r.t. Y (proposition 4.1 of [lam2026a])
# ---------------------------------------------------------------------------

def grad_L_alpha(X: Array, Y: Array,
                 DX: Array, DY: Array,
                 max_pairs: int) -> Array:
    """
    d L_alpha / d y_k = -(4/N^2) sum_j sign(DX[k,j]-DY[k,j]) * (y_k - y_j)

    for large N uses stochastic approximation over max_pairs sampled pairs.
    """
    N, d = Y.shape
    G = np.zeros((N, d))
    c1 = 4.0 / (N * N)

    if N * (N-1) <= max_pairs:
        # exact
        S = np.sign(DX - DY)   # (N, N)
        np.fill_diagonal(S, 0.0)
        # G[k] = -c1 * sum_j S[k,j] * (y_k - y_j)
        #       = -c1 * (S @ Y diag - ...) -- expand:
        # sum_j S[k,j]*(y_k-y_j) = (sum_j S[k,j])*y_k - sum_j S[k,j]*y_j
        row_sum = S.sum(axis=1, keepdims=True)   # (N,1)
        G = -c1 * (row_sum * Y - S @ Y)
    else:
        # stochastic: sample max_pairs ordered pairs (k, j), k != j
        scale = float(N * (N-1)) / max_pairs
        rng = np.random.default_rng()
        ks = rng.integers(0, N, size=max_pairs)
        js = rng.integers(0, N-1, size=max_pairs)
        js = np.where(js >= ks, js + 1, js)
        sg = np.sign(DX[ks, js] - DY[ks, js])
        diff = Y[ks] - Y[js]
        np.add.at(G, ks, -c1 * scale * (sg[:, None] * diff))
        # symmetric contribution to j
        np.add.at(G, js,  c1 * scale * (sg[:, None] * diff))

    return G


# ---------------------------------------------------------------------------
# one gradient step
# ---------------------------------------------------------------------------

def gd_step(X: Array, Y: Array, DX: Array, cfg: Config,
            rng: np.random.Generator) -> dict:
    """
    one step of projected gradient descent on J(theta).

    returns dict with loss values and updated Y.
    """
    idx, sk = build_knn(Y, cfg.k)
    lt, wp, wm = log_L_tilde(sk, cfg.tau)
    DY = sq_dist_matrix(Y)
    la = L_alpha(DX, DY)
    lb = L_beta(sk)
    loss = la + cfg.lam * lt

    G_la = grad_L_alpha(X, Y, DX, DY, cfg.max_grad_pairs)
    G_lb = grad_log_L_tilde(Y, idx, sk, wp, wm, cfg.k)

    G = G_la + cfg.lam * G_lb

    # per-point gradient clipping
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    mask  = norms.ravel() > cfg.grad_clip
    G[mask] *= cfg.grad_clip / norms[mask]

    Y_new = np.clip(Y - cfg.eta * G, 0.02, 0.98)

    return dict(Y=Y_new, la=la, lb=lb, lt=lt, loss=loss,
                mk=sk.min(), Mk=sk.max())


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(X: Array, cfg: Config = Config()) -> dict:
    """
    run n_iter steps of gradient descent on J.

    X: (N, d) input cloud in [0,1]^d.

    returns dict with:
      Y       -- trained cloud T_theta(X)
      history -- list of per-step dicts {la, lb, lt, loss, mk, Mk}
      mk, Mk  -- bracket endpoints of trained Y
    """
    rng = np.random.default_rng(cfg.seed)
    N = len(X)

    # initialize Y near X
    Y = X + rng.normal(0.0, 1e-3, size=X.shape)
    Y = np.clip(Y, 0.02, 0.98)

    DX = sq_dist_matrix(X)

    # initial bracket
    _, sk0 = build_knn(X, cfg.k)
    R0  = L_beta(sk0)
    mk0 = sk0.min()

    history = []
    for step in range(cfg.n_iter):
        result = gd_step(X, Y, DX, cfg, rng)
        Y = result['Y']
        history.append({k: v for k, v in result.items() if k != 'Y'})

        if (step+1) % 20 == 0 or step == 0:
            r = history[-1]
            print(f"iter {step+1:4d}  L_alpha={r['la']:.5f}"
                  f"  L_beta={r['lb']:.4f}  J={r['loss']:.5f}")

    _, sk_final = build_knn(Y, cfg.k)

    return dict(
        Y       = Y,
        history = history,
        mk      = float(sk_final.min()),
        Mk      = float(sk_final.max()),
        R0      = float(R0),
        mk0     = float(mk0),
    )


# ---------------------------------------------------------------------------
# defect lower bound (corollary 3.3 of [lam2026a])
# ---------------------------------------------------------------------------

def defect_lower_bound(k: int, mk0: float, N: int, R0: float,
                        R_star: float) -> float:
    """
    lower bound on L_alpha achievable by any algorithm that reduces
    L_beta^{(k)} from R0 to R_star.

        L_alpha >= k * mk0 / ((1 + R*) * N^2) * (R0 - R*)

    R_star must satisfy 1 <= R_star <= R0.
    """
    if not (1.0 <= R_star <= R0):
        raise ValueError("R_star must be in [1, R0]")
    return k * mk0 / ((1.0 + R_star) * N**2) * (R0 - R_star)


# ---------------------------------------------------------------------------
# example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    N = 200

    # two gaussian cloud: dense left cluster + sparse right
    n1 = round(N * 0.55)
    X = np.vstack([
        rng.normal([0.28, 0.52], 0.09, (n1, 2)),
        rng.normal([0.70, 0.42], 0.14, (N - n1, 2)),
    ])
    X = np.clip(X, 0.02, 0.98)

    cfg = Config(k=5, lam=0.3, tau=8, eta=0.005, n_iter=100, seed=0)
    result = train(X, cfg)

    Y       = result['Y']
    history = result['history']

    print(f"\nR0     = {result['R0']:.4f}  (initial L_beta)")
    print(f"L_beta = {history[-1]['lb']:.4f}  (final)")
    print(f"mk     = {result['mk']:.6f}")
    print(f"Mk     = {result['Mk']:.6f}")

    lb_curve = [h['lb'] for h in history]
    la_curve = [h['la'] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(X[:, 0], X[:, 1], s=12, c='grey', label='X (raw)')
    axes[0].scatter(Y[:, 0], Y[:, 1], s=12, c='steelblue', label='Y = T(X)')
    axes[0].set_title('X and Y')
    axes[0].legend(fontsize=8)

    axes[1].plot(la_curve, color='firebrick', label='L_alpha')
    axes[1].plot(lb_curve, color='purple',    label='L_beta^(k)')
    axes[1].set_xlabel('iteration')
    axes[1].legend(fontsize=8)
    axes[1].set_title('loss curves')

    # pareto plane
    axes[2].plot(la_curve, lb_curve, color='purple', lw=1.5, label='trajectory')
    R0 = result['R0']; mk0 = result['mk0']
    R_vals = np.linspace(1.0, R0, 200)
    lb_bound = [defect_lower_bound(cfg.k, mk0, N, R0, r) for r in R_vals]
    axes[2].plot(lb_bound, R_vals, '--', color='goldenrod', label='defect lower bound')
    axes[2].set_xlabel('L_alpha')
    axes[2].set_ylabel('L_beta^(k)')
    axes[2].legend(fontsize=8)
    axes[2].set_title('pareto plane')

    plt.tight_layout()
    plt.savefig('algo1_result.png', dpi=150)
    plt.show()
