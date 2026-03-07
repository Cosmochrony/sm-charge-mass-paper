import math
import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm

def build_blocks(k_max: int):
    blocks = []
    idx = 0
    for k in range(k_max + 1):
        dim = (k + 1) ** 2
        fiber = idx
        base = list(range(idx + 1, idx + dim))
        blocks.append({"k": k, "start": idx, "dim": dim, "fiber": fiber, "base": base})
        idx += dim
    return blocks, idx

def delta_operator(blocks, n):
    d = np.zeros(n)
    for b in blocks:
        k = b["k"]
        lam = k * (k + 2)
        d[b["start"] : b["start"] + b["dim"]] = lam
    return np.diag(d)

def unit_vec_real(n, rng):
    v = rng.normal(size=n)
    v = v / norm(v)
    return v

def omega2_hopf(blocks, n, r, eps=1.0, j_intra=0.35, j_inter=0.25, seed=0):
    rng = np.random.default_rng(seed)
    om = np.zeros((n, n), dtype=float)

    for b in blocks:
        f = b["fiber"]
        om[f, f] += eps
        for bi in b["base"]:
            om[bi, bi] += -eps / r

        if b["dim"] > 1 and j_intra != 0.0:
            v = unit_vec_real(len(b["base"]), rng)
            for a, idxb in zip(v, b["base"]):
                om[f, idxb] += j_intra * a
                om[idxb, f] += j_intra * a

    for i in range(len(blocks) - 1):
        b, bp = blocks[i], blocks[i + 1]
        f, fp = b["fiber"], bp["fiber"]
        om[f, fp] += j_inter
        om[fp, f] += j_inter

        if bp["dim"] > 1:
            v = unit_vec_real(len(bp["base"]), rng)
            for a, idxb in zip(v, bp["base"]):
                om[f, idxb] += 0.6 * j_inter * a
                om[idxb, f] += 0.6 * j_inter * a

    return 0.5 * (om + om.T)

def add_noise_sym(om, sigma, rng):
    if sigma == 0.0:
        return om
    a = rng.normal(size=om.shape)
    h = 0.5 * (a + a.T)
    h = h / norm(h, "fro")
    return om + sigma * h

def projector_on_index(n, idx):
    p = np.zeros((n, n), dtype=float)
    p[idx, idx] = 1.0
    return p

def time_series_expectation(om, p, psi0, t_max=180.0, dt=0.2):
    t = np.arange(0.0, t_max, dt)
    vals = np.empty_like(t)

    # brute expm is OK for n ~ 100-250 and small dt on a laptop
    for i, ti in enumerate(t):
        u = expm(-1j * ti * om)
        psi = u @ psi0
        vals[i] = float(np.real(np.vdot(psi, p @ psi)))
    return t, vals

def fft_metrics(vals, dt, top_k=6):
    x = vals - np.mean(vals)
    win = np.hanning(len(x))
    X = np.fft.rfft(x * win)
    p = np.abs(X) ** 2
    p = p / np.sum(p)

    top = np.sort(p)[-top_k:]
    sharp = float(np.sum(top))
    ent = float(-np.sum(p * np.log(p + 1e-15)))
    return sharp, ent

def run_scan(k_max=6, r_list=None, sigmas=None, trials=8, seed=1):
    if r_list is None:
        r_list = np.linspace(1.2, 2.4, 25)
    if sigmas is None:
        sigmas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]

    blocks, n = build_blocks(k_max)
    _delta = delta_operator(blocks, n)

    # P_parallel: fiber mode in k=1 block
    f1 = blocks[1]["fiber"]
    p_par = projector_on_index(n, f1)

    # initial state: fiber(k=1) + random base direction in same multiplet
    rng = np.random.default_rng(seed)
    v = unit_vec_real(len(blocks[1]["base"]), rng)
    psi0 = np.zeros(n, dtype=complex)
    psi0[f1] = 1.0
    for a, idxb in zip(v, blocks[1]["base"]):
        psi0[idxb] += a
    psi0 = psi0 / norm(psi0)

    out = {}
    for r in r_list:
        om0 = omega2_hopf(blocks, n, r, seed=seed + 7)
        comm = _delta @ om0 - om0 @ _delta
        comm_norm = float(norm(comm, "fro"))

        out[r] = {"comm_norm": comm_norm, "by_sigma": {}}
        for s in sigmas:
            sharp_list = []
            ent_list = []
            for tr in range(trials):
                rngt = np.random.default_rng(seed + 1000 * tr + 3)
                om = add_noise_sym(om0, s, rngt)
                _, vals = time_series_expectation(om, p_par, psi0, t_max=160.0, dt=0.2)
                sharp, ent = fft_metrics(vals, 0.2, top_k=6)
                sharp_list.append(sharp)
                ent_list.append(ent)

            out[r]["by_sigma"][s] = {
                "sharp": float(np.mean(sharp_list)),
                "entropy": float(np.mean(ent_list)),
            }

    return out

if __name__ == "__main__":
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    r_list = np.array([1.35, 1.5, phi, 1.8, 2.1])
    res = run_scan(k_max=6, r_list=r_list, sigmas=[0.0, 0.05, 0.1, 0.2], trials=6, seed=4)
    for r in r_list:
        print(f"r={r:.6f}  comm_norm={res[r]['comm_norm']:.3e}")
        for s, m in res[r]["by_sigma"].items():
            print(f"  sigma={s:.2f}  sharp={m['sharp']:.3f}  entropy={m['entropy']:.3f}")