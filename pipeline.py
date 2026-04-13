"""
pipeline.py

full generation pipeline from a raw labeled point cloud X.

phase 1: algorithm 1 (algo1_train.py)
  train T_theta on
      J(theta) = L_alpha(X, Y) + lambda * log L_tilde_{beta,tau}^{(k)}(Y)
  where Y = T_theta(X).
  gradient descent reduces L_beta^{(k)}(Y) toward 1 while keeping
  L_alpha(X, Y) small, enforcing close-to-geometry-preserving transport.

phase 2: algorithm 2 (algo2_generate.py)
  sample from the trained cloud Y using the certified barycentric filter.
  each accepted y_hat satisfies m_k <= S_k(y_hat; Y) <= M_k.
  invert via T_theta^{-1} to recover x_hat in the original space.

the inverse map T_theta^{-1} is approximated here by nearest-neighbor
interpolation on the pairs (y_i, x_i).  for a differentiable T_theta
(e.g. a neural network or spline map) this would be the exact analytic
inverse or a fixed-point iteration.

reference: [lam2026a] close-to-geometry-preserving sampleable generation
from labeled point clouds.

hy p. g. lam
worcester polytechnic institute
hlam@wpi.edu
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

from algo1_train  import train,    Config   as TrainConfig
from algo2_generate import generate, compute_bracket


Array = np.ndarray


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # algo 1 parameters
    k:         int   = 5
    lam:       float = 0.3
    tau:       int   = 8
    eta:       float = 0.005
    n_iter:    int   = 200
    grad_clip: float = 0.08
    max_grad_pairs: int = 20_000

    # algo 2 parameters
    n_samples:  int  = 200
    max_trials: int  = 10_000

    seed: Optional[int] = 0


# ---------------------------------------------------------------------------
# nearest-neighbor inverse map
# ---------------------------------------------------------------------------

def nn_inverse(Y: Array, X: Array) -> callable:
    """
    approximate T_theta^{-1}(p) by the x_i whose y_i is closest to p.
    valid when T_theta is close to an isometry (L_alpha small).
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Y)

    def T_inv(p: Array) -> Array:
        _, idx = nn.kneighbors(p[None, :])
        return X[idx[0, 0]]

    return T_inv


# ---------------------------------------------------------------------------
# run full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(X: Array, cfg: PipelineConfig = PipelineConfig()) -> dict:
    """
    full pipeline: train T_theta on X, then generate certified samples.

    X: (N, d) raw labeled cloud.

    returns dict with
      Y           -- trained cloud T_theta(X)
      train_result-- full training output dict
      gen_result  -- full generation output dict
      x_samples   -- accepted samples in X-space via T_theta^{-1}
      y_samples   -- accepted samples in Y-space
    """
    N = len(X)

    # phase 1: train T_theta
    print(f"phase 1: training T_theta on N={N} points, k={cfg.k}, "
          f"lambda={cfg.lam}, n_iter={cfg.n_iter}")

    train_cfg = TrainConfig(
        k              = cfg.k,
        lam            = cfg.lam,
        tau            = cfg.tau,
        eta            = cfg.eta,
        n_iter         = cfg.n_iter,
        grad_clip      = cfg.grad_clip,
        max_grad_pairs = cfg.max_grad_pairs,
        seed           = cfg.seed,
    )
    train_result = train(X, train_cfg)
    Y = train_result['Y']

    lb_final = train_result['history'][-1]['lb']
    la_final = train_result['history'][-1]['la']
    mk = train_result['mk']
    Mk = train_result['Mk']

    print(f"  L_beta final = {lb_final:.4f}  (started at R0={train_result['R0']:.4f})")
    print(f"  L_alpha final= {la_final:.5f}")
    print(f"  bracket: mk={mk:.6f}  Mk={Mk:.6f}  ratio={Mk/mk:.3f}")

    # phase 2: barycentric generation
    print(f"\nphase 2: generating {cfg.n_samples} certified samples "
          f"(max trials={cfg.max_trials})")

    T_inv   = nn_inverse(Y, X)
    gen_result = generate(
        Y          = Y,
        k          = cfg.k,
        n_samples  = cfg.n_samples,
        T_inv      = T_inv,
        max_trials = cfg.max_trials,
        seed       = cfg.seed,
    )

    print(f"  trials={gen_result['trials']}  "
          f"accepted={gen_result['accepted']}  "
          f"accept rate={gen_result['accept_rate']:.3f}")

    return dict(
        Y            = Y,
        train_result = train_result,
        gen_result   = gen_result,
        y_samples    = gen_result['samples'],
        x_samples    = gen_result['x_samples'],
    )


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_pipeline(X: Array, result: dict,
                  figpath: Optional[str] = None) -> None:
    """
    four-panel figure:
      (a) raw X
      (b) trained Y with S_k heatmap
      (c) loss curves from training
      (d) Y with accepted samples overlaid
    """
    Y        = result['Y']
    history  = result['train_result']['history']
    y_samp   = result['y_samples']
    x_samp   = result['x_samples']
    gen      = result['gen_result']

    nbr_idx, sk, mk, Mk = compute_bracket(Y, len(nbr_idx := np.empty(0)))
    # recompute sk properly
    from sklearn.neighbors import NearestNeighbors
    # infer k from history context: use 5 as default if not stored
    # (in practice pass k explicitly or store it in result)
    # here we read it from the train bracket
    mk_val = result['train_result']['mk']
    Mk_val = result['train_result']['Mk']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # (a) raw cloud
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=10, c='grey')
    ax.set_title('(a)  raw cloud X')
    ax.set_aspect('equal')

    # (b) trained cloud colored by sk
    ax = axes[1]
    _, sk_Y, _, _ = compute_bracket(Y, result['train_result']['history'][0].get('k', 5))
    sc = ax.scatter(Y[:, 0], Y[:, 1], s=10,
                    c=sk_Y, cmap='coolwarm')
    plt.colorbar(sc, ax=ax, label='S_k(i;Y)')
    ax.set_title('(b)  trained Y (color = S_k)')
    ax.set_aspect('equal')

    # (c) loss curves
    ax = axes[2]
    la_c = [h['la'] for h in history]
    lb_c = [h['lb'] for h in history]
    ax.plot(la_c, color='firebrick', label='L_alpha')
    ax2 = ax.twinx()
    ax2.plot(lb_c, color='purple', label='L_beta^(k)')
    ax.set_xlabel('iteration')
    ax.set_ylabel('L_alpha', color='firebrick')
    ax2.set_ylabel('L_beta^(k)', color='purple')
    ax.set_title('(c)  training loss curves')
    lines = [plt.Line2D([0],[0],color='firebrick'), plt.Line2D([0],[0],color='purple')]
    ax.legend(lines, ['L_alpha','L_beta^(k)'], fontsize=8)

    # (d) generated samples
    ax = axes[3]
    ax.scatter(Y[:, 0], Y[:, 1], s=8, c='steelblue', alpha=0.35, label='Y')
    if len(y_samp) > 0:
        ax.scatter(y_samp[:, 0], y_samp[:, 1], s=20, marker='D',
                   c='seagreen', zorder=3,
                   label=f'{gen["accepted"]} samples (rate={gen["accept_rate"]:.2f})')
    ax.set_title('(d)  algorithm 2 output')
    ax.legend(fontsize=7)
    ax.set_aspect('equal')

    plt.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=150)
        print(f"figure saved to {figpath}")
    plt.show()


# ---------------------------------------------------------------------------
# cloud generators
# ---------------------------------------------------------------------------

def _gauss(rng, mx, my, sx, sy, n):
    pts = rng.normal([mx, my], [sx, sy], (n, 2))
    return np.clip(pts, 0.03, 0.97)


def make_cloud(name: str, N: int, seed: int = 0) -> Array:
    """
    generate a 2d point cloud in [0,1]^2 by name.
    available: two_gauss, tendrils, annulus, figure8, horseshoe,
               spiral, concentric, two_moons, swiss_roll.
    """
    rng = np.random.default_rng(seed)

    def clp(x): return np.clip(x, 0.03, 0.97)

    if name == 'two_gauss':
        n1 = round(N * 0.55)
        return np.vstack([_gauss(rng, 0.28, 0.52, 0.09, 0.09, n1),
                          _gauss(rng, 0.70, 0.42, 0.14, 0.14, N - n1)])

    elif name == 'tendrils':
        a = []
        nHub = round(N * 0.22)
        a.append(_gauss(rng, 0.45, 0.5, 0.04, 0.04, nHub))
        for frac, cx, cy, dx, dy in [
            (0.28, 0.45, 0.50, 0.42, -0.22),
            (0.20, 0.45, 0.50, 0.26,  0.28),
            (0.18, 0.45, 0.50,-0.28,  0.08),
        ]:
            n_ = round(N * frac)
            t  = rng.random(n_)
            a.append(np.c_[clp(cx + t*dx + rng.normal(0,.02,n_)),
                           clp(cy + t*dy + rng.normal(0,.02,n_))])
        n4 = N - sum(len(v) for v in a)
        t  = rng.random(n4)
        a.append(np.c_[clp(0.45 - t*0.22 + t**2*0.15 + rng.normal(0,.02,n4)),
                       clp(0.50 - t*0.26              + rng.normal(0,.02,n4))])
        return np.vstack(a)[:N]

    elif name == 'annulus':
        pts = []
        while len(pts) < N:
            r = 0.20 + rng.random() * 0.08
            t = rng.random() * 2 * np.pi
            if rng.random() < (1 + 0.7*abs(np.cos(t))) / 1.7:
                pts.append([clp(0.5 + r*np.cos(t)), clp(0.5 + r*np.sin(t))])
        return np.array(pts[:N])

    elif name == 'figure8':
        t = rng.random(N) * 2 * np.pi
        sc = 0.22 / (1 + np.sin(t)**2)
        return np.c_[clp(0.5 + sc*np.cos(t) + rng.normal(0,.018,N)),
                     clp(0.5 + sc*np.sin(t)*np.cos(t) + rng.normal(0,.018,N))]

    elif name == 'horseshoe':
        nL = round(N * 0.35)
        nB = N - 2*nL
        t  = rng.random(nB)
        return np.vstack([
            _gauss(rng, 0.22, 0.65, 0.04, 0.09, nL),
            _gauss(rng, 0.78, 0.65, 0.04, 0.09, nL),
            np.c_[clp(0.22 + t*0.56 + rng.normal(0,.04,nB)),
                  clp(0.28          + rng.normal(0,.06,nB))],
        ])

    elif name == 'spiral':
        t  = rng.random(N)**0.6
        r  = 0.06 + t*0.40
        th = t * 3.5 * 2 * np.pi
        return np.c_[clp(0.5 + r*np.cos(th) + rng.normal(0,.014,N)),
                     clp(0.5 + r*np.sin(th) + rng.normal(0,.014,N))]

    elif name == 'concentric':
        pts = []
        for rv, rw, frac in [(0.08,0.018,0.30),(0.18,0.022,0.25),(0.34,0.030,0.45)]:
            nr = round(N * frac)
            t  = rng.random(nr) * 2 * np.pi
            dr = (rng.random(nr) - 0.5) * rw
            pts.append(np.c_[clp(0.5 + (rv+dr)*np.cos(t)),
                              clp(0.5 + (rv+dr)*np.sin(t))])
        return np.vstack(pts)[:N]

    elif name == 'two_moons':
        h = N // 2
        t0 = np.linspace(0, np.pi, h)
        t1 = np.linspace(np.pi, 2*np.pi, N-h)
        top = np.c_[clp(0.25+0.36*np.cos(t0)+rng.normal(0,.028,h)),
                    clp(0.38+0.30*np.sin(t0)+rng.normal(0,.028,h))]
        bot = np.c_[clp(0.75+0.36*np.cos(t1)+rng.normal(0,.028,N-h)),
                    clp(0.62+0.30*np.sin(t1)+rng.normal(0,.028,N-h))]
        return np.vstack([top, bot])

    elif name == 'swiss_roll':
        t = 1.5*np.pi*(1 + 2*rng.random(N))
        w = rng.random(N)
        return np.c_[clp((t*np.cos(t)/(9*np.pi)+0.55)*0.88+0.06+rng.normal(0,.02,N)),
                     clp(w*0.86+0.07+rng.normal(0,.02,N))]

    else:
        raise ValueError(f"unknown cloud name: {name}")


# ---------------------------------------------------------------------------
# example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    cloud_name = sys.argv[1] if len(sys.argv) > 1 else 'two_gauss'
    N          = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print(f"cloud: {cloud_name}  N={N}")

    X = make_cloud(cloud_name, N, seed=0)

    cfg = PipelineConfig(
        k          = 5,
        lam        = 0.3,
        tau        = 8,
        eta        = 0.005,
        n_iter     = 150,
        n_samples  = 200,
        max_trials = 10_000,
        seed       = 0,
    )

    result = run_pipeline(X, cfg)

    # minimal plot without the helper (avoid recomputing bracket)
    Y      = result['Y']
    y_samp = result['y_samples']
    hist   = result['train_result']['history']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].scatter(X[:, 0], X[:, 1], s=10, c='grey')
    axes[0].set_title('raw X')
    axes[0].set_aspect('equal')

    axes[1].scatter(Y[:, 0], Y[:, 1], s=10, c='steelblue', alpha=0.5, label='Y')
    if len(y_samp):
        axes[1].scatter(y_samp[:, 0], y_samp[:, 1], s=18, marker='D',
                        c='seagreen', label='samples')
    axes[1].legend(fontsize=8)
    axes[1].set_title('Y and accepted samples')
    axes[1].set_aspect('equal')

    axes[2].plot([h['la'] for h in hist], color='firebrick', label='L_alpha')
    axes[2].plot([h['lb'] for h in hist], color='purple',    label='L_beta^(k)')
    axes[2].set_xlabel('iteration')
    axes[2].legend(fontsize=8)
    axes[2].set_title('training curves')

    plt.suptitle(f'pipeline: {cloud_name}  N={N}', fontsize=11)
    plt.tight_layout()
    plt.savefig('pipeline_result.png', dpi=150)
    plt.show()
