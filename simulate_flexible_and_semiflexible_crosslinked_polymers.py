#!/usr/bin/env python3
"""
Simulation of a flexible polymer (subsystem A) crosslinked to a semiflexible
polymer (subsystem B), as used in

    C. Lim and J.-H. Jeon, "..." (Phys. Rev. Research, 2025),
    arXiv:2507.08291

Both polymers are modeled via coupled Langevin equations on the GPU (CuPy).
The code tracks the central monomer of A and B and computes their MSDs,
combining multiple sampling periods into a single MSD curve.

Usage (example):

    python simulate_flexible_and_semiflexible_crosslinked_polymers.py \\
        --k_A 500 --k_spring 100 --T 1000000 --gpu 0 \\
        --prefix msd_flexA_semiB --DIR ./data/

Outputs:

    <DIR>/<prefix>_kA<k_A>_kS<k_spring>.npz
        - tLags : lag times (physical time)
        - msd_A : MSD of central bead of polymer A
        - msd_B : MSD of central bead of polymer B
"""

import os
import argparse
from typing import Dict, Tuple

import numpy as np
import cupy as cp
from tqdm import trange


# --------------------------------------------------------------------------- #
# MSD utilities (same logic as in the first script; kept local for simplicity)
# --------------------------------------------------------------------------- #
def calc_MSD_cp(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MSD from a trajectory array using CuPy for speed.

    Parameters
    ----------
    traj : ndarray, shape (T, Nens, 3)

    Returns
    -------
    tLags : ndarray
        Lag steps.
    MSD : ndarray
        MSD(tLag).
    """
    n_frames = len(traj)
    n_log = int(np.log10(n_frames)) * 40 if n_frames > 10 else 10
    tLags = np.unique(
        np.logspace(0, np.log10(n_frames), n_log).astype(int)
    )[:-1]

    traj_cp = cp.array(traj)
    MSD = np.zeros(len(tLags), dtype=float)

    for i, tLag in enumerate(tLags):
        MSD[i] = float(
            cp.nanmean((traj_cp[tLag:] - traj_cp[:-tLag]) ** 2) * 3.0
        )
    return tLags, MSD


def _valid_tail_rows(traj_np: np.ndarray) -> np.ndarray:
    """Return only the valid tail (front is NaN-padded)."""
    row_ok = np.any(
        np.isfinite(traj_np).reshape(traj_np.shape[0], -1),
        axis=1,
    )
    if not np.any(row_ok):
        return traj_np[0:0]
    first_valid = np.argmax(row_ok)
    return traj_np[first_valid:]


def concat_msds_with_mask(
    traj_dict: Dict[int, np.ndarray],
    Tperiods,
    dt: float,
    burning_ratio: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine MSDs from trajectories sampled at different periods.

    Parameters
    ----------
    traj_dict : dict
        {p: (MAX_KEEP, Nens, 3)} with NaN padding.
    Tperiods : list[int]
    dt : float
    burning_ratio : int

    Returns
    -------
    t_all, msd_all : ndarray
    """
    periods_sorted = sorted(Tperiods)
    t_all_list = []
    msd_all_list = []

    for i, p in enumerate(periods_sorted):
        traj_p = _valid_tail_rows(traj_dict[p])
        if traj_p.shape[0] < 2:
            continue

        tLags, MSD = calc_MSD_cp(traj_p)

        if i < len(periods_sorted) - 1:
            p_next = periods_sorted[i + 1]
            tLag_max = burning_ratio * (p_next / p)
        else:
            tLag_max = np.inf

        if i > 0:
            mask = (tLags >= burning_ratio) & (tLags < tLag_max)
        else:
            mask = (tLags < tLag_max)
        if not np.any(mask):
            continue

        tLags_sel = tLags[mask]
        MSD_sel = MSD[mask]

        steps = tLags_sel.astype(np.int64) * int(p)
        t_phys = steps * dt

        t_all_list.append(t_phys)
        msd_all_list.append(MSD_sel.astype(float))

    if not t_all_list:
        return np.array([]), np.array([])

    t_all = np.concatenate(t_all_list)
    msd_all = np.concatenate(msd_all_list)
    order = np.argsort(t_all)
    return t_all[order], msd_all[order]


# --------------------------------------------------------------------------- #
# Main simulation
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flexible polymer A crosslinked to semiflexible polymer B."
    )
    parser.add_argument(
        "--k_A",
        type=float,
        default=50 * 10,
        help="Spring constant k_A for system A (default: 500).",
    )
    parser.add_argument(
        "--k_spring",
        type=float,
        default=100.0,
        help="Coupling spring constant between A and B (default: 100).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="msd",
        help="Prefix for output file (default: 'msd').",
    )
    parser.add_argument(
        "--DIR",
        type=str,
        default="./data/",
        help="Directory to save output (default: ./data/).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID for CuPy (default: 0).",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=int(1e6),
        help="Number of time steps to simulate (default: 1e6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    k_A = float(args.k_A)
    k_spring = float(args.k_spring)
    T = int(args.T)

    os.makedirs(args.DIR, exist_ok=True)
    cp.cuda.runtime.setDevice(int(args.gpu))

    # ------------------------------------------------------------------ #
    # Model parameters
    # ------------------------------------------------------------------ #
    # A: flexible
    kappa_A = 0.0
    lamb_M_A = (1 + kappa_A**2) / (1 - kappa_A**2) / 2.0
    lamb_E_A = 1.0 / (1 - kappa_A**2) / 2.0
    mu_A = kappa_A / (1 - kappa_A**2)
    mu_2_A = mu_A / 2.0
    gamma_A = k_A * 0.01  # sets initial timescale

    # B: semiflexible
    kappa_B = 0.995
    lamb_M_B = (1 + kappa_B**2) / (1 - kappa_B**2) / 2.0
    lamb_E_B = 1.0 / (1 - kappa_B**2) / 2.0
    mu_B = kappa_B / (1 - kappa_B**2)
    mu_2_B = mu_B / 2.0
    k_B = 1.0
    gamma_B = 2.0

    # Optional: persistence length of B in model units
    L_p_B = 1.0 / (1.0 - kappa_B)  # kept for reference, not used further

    # ------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------ #
    N_A = 1000  # beads on each side of central bead in A
    N_B = 50    # beads on each side of central bead in B
    Nens = 1000

    coords_A = cp.zeros((Nens, 2 * N_A + 1, 3))
    coords_B = cp.zeros((Nens, 2 * N_B + 1, 3))

    M_A = 2 * N_A
    M_B = 2 * N_B

    # Equilibrium statistics for A
    COV_A = np.zeros((M_A, M_A), dtype=float)
    rho_A = kappa_A / k_A if k_A != 0 else 0.0

    if kappa_A == 0.0:
        for i in range(M_A):
            COV_A[i, i] = 1.0 / k_A
        for i in range(Nens):
            coords_A[i, 1:] = cp.random.normal(
                0.0, np.sqrt(1.0 / k_A), size=(M_A, 3)
            ).cumsum(axis=1)
    else:
        for i in range(M_A):
            for j in range(M_A):
                COV_A[i, j] = rho_A ** abs(i - j)
        for i in range(Nens):
            coords_A[i, 1:] = cp.random.multivariate_normal(
                np.zeros(M_A), COV_A, size=3
            ).cumsum(axis=1).T

    # Equilibrium statistics for B
    COV_B = np.zeros((M_B, M_B), dtype=float)
    rho_B = kappa_B / k_B if k_B != 0 else 0.0
    for i in range(M_B):
        for j in range(M_B):
            COV_B[i, j] = rho_B ** abs(i - j)
    if kappa_B == 0.0:
        for i in range(M_B):
            COV_B[i, i] = 1.0 / k_B

    for i in range(Nens):
        coords_B[i, 1:] = cp.random.multivariate_normal(
            np.zeros(M_B), COV_B, size=3
        ).cumsum(axis=1).T

    # Align the central bead of B with the central bead of A
    coords_B += coords_A[:, N_A][:, None] - coords_B[:, N_B][:, None]
    if k_spring != 0.0:
        coords_B += cp.random.normal(
            0.0, np.sqrt(1.0 / k_spring), Nens
        )[:, None, None]

    # ------------------------------------------------------------------ #
    # Simulation config
    # ------------------------------------------------------------------ #
    dt = 1e-3

    Tperiods = [1, 10, 100, 1000, 10000]
    MAX_KEEP = 10_000

    traj_A_dict = {
        p: np.full((MAX_KEEP, Nens, 3), np.nan, dtype=float) for p in Tperiods
    }
    traj_B_dict = {
        p: np.full((MAX_KEEP, Nens, 3), np.nan, dtype=float) for p in Tperiods
    }

    n_save = {p: min(T // p, MAX_KEEP) for p in Tperiods}
    t_start = {p: T - p * n_save[p] for p in Tperiods}
    save_idx = {p: 0 for p in Tperiods}

    # ------------------------------------------------------------------ #
    # GPU working arrays
    # ------------------------------------------------------------------ #
    a_n_dt_A = cp.zeros((Nens, 2 * N_A + 1, 3))
    k_n_dt_A = cp.zeros((Nens, 2 * N_A + 1, 3))

    a_n_dt_B = cp.zeros((Nens, 2 * N_B + 1, 3))
    k_n_dt_B = cp.zeros((Nens, 2 * N_B + 1, 3))

    sqrt2dt_A = np.sqrt(2.0 * dt / gamma_A)
    sqrt2dt_over3_A = np.sqrt(2.0 * dt / (3.0 * gamma_A))
    sqrt2dt_B = np.sqrt(2.0 * dt / gamma_B)
    sqrt2dt_over3_B = np.sqrt(2.0 * dt / (3.0 * gamma_B))

    # ------------------------------------------------------------------ #
    # Time integration loop
    # ------------------------------------------------------------------ #
    for t in trange(
        T, mininterval=5, desc="Simulating (flex A + semiflex B)"
    ):
        # Save positions of central beads
        for p in Tperiods:
            if (t >= t_start[p]) and ((t - t_start[p]) % p == 0):
                i = save_idx[p]
                if i < n_save[p]:
                    dst = MAX_KEEP - n_save[p] + i
                    traj_A_dict[p][dst] = coords_A[:, N_A].get()
                    traj_B_dict[p][dst] = coords_B[:, N_B].get()
                    save_idx[p] += 1

        # ===== System A =====
        dW_A = cp.random.normal(
            0.0, 1.0, (Nens, 2 * N_A + 1, 3)
        ) * sqrt2dt_A
        dH_A = cp.random.normal(
            0.0, 1.0, (Nens, 2 * N_A + 1, 3)
        ) * sqrt2dt_over3_A

        a_n_dt_A[:] = 0.0
        # boundaries
        a_n_dt_A[:, 0] = (
            -lamb_E_A * coords_A[:, 0]
            + (lamb_E_A + mu_2_A) * coords_A[:, 1]
            - mu_2_A * coords_A[:, 2]
        )
        a_n_dt_A[:, 1] = (
            (lamb_E_A + mu_2_A) * coords_A[:, 0]
            + (-lamb_E_A - lamb_M_A - mu_A) * coords_A[:, 1]
            + (lamb_M_A + mu_A) * coords_A[:, 2]
            - mu_2_A * coords_A[:, 3]
        )
        # interior
        a_n_dt_A[:, 2:-2] = (
            -mu_2_A * coords_A[:, :-4]
            + (lamb_M_A + mu_A) * coords_A[:, 1:-3]
            + (-2.0 * lamb_M_A - mu_A) * coords_A[:, 2:-2]
            + (lamb_M_A + mu_A) * coords_A[:, 3:-1]
            - mu_2_A * coords_A[:, 4:]
        )
        # boundaries
        a_n_dt_A[:, -2] = (
            -mu_2_A * coords_A[:, -4]
            + (lamb_M_A + mu_A) * coords_A[:, -3]
            + (-lamb_E_A - lamb_M_A - mu_A) * coords_A[:, -2]
            + (lamb_E_A + mu_2_A) * coords_A[:, -1]
        )
        a_n_dt_A[:, -1] = (
            -mu_2_A * coords_A[:, -3]
            + (lamb_E_A + mu_2_A) * coords_A[:, -2]
            - lamb_E_A * coords_A[:, -1]
        )

        a_n_dt_A *= 2.0 * k_A / gamma_A * dt
        a_n_dt_A[:, N_A] += (
            -k_spring / gamma_A * (coords_A[:, N_A] - coords_B[:, N_B]) * dt
        )

        # ===== System B =====
        dW_B = cp.random.normal(
            0.0, 1.0, (Nens, 2 * N_B + 1, 3)
        ) * sqrt2dt_B
        dH_B = cp.random.normal(
            0.0, 1.0, (Nens, 2 * N_B + 1, 3)
        ) * sqrt2dt_over3_B

        a_n_dt_B[:] = 0.0
        # boundaries
        a_n_dt_B[:, 0] = (
            -lamb_E_B * coords_B[:, 0]
            + (lamb_E_B + mu_2_B) * coords_B[:, 1]
            - mu_2_B * coords_B[:, 2]
        )
        a_n_dt_B[:, 1] = (
            (lamb_E_B + mu_2_B) * coords_B[:, 0]
            + (-lamb_E_B - lamb_M_B - mu_B) * coords_B[:, 1]
            + (lamb_M_B + mu_B) * coords_B[:, 2]
            - mu_2_B * coords_B[:, 3]
        )
        # interior
        a_n_dt_B[:, 2:-2] = (
            -mu_2_B * coords_B[:, :-4]
            + (lamb_M_B + mu_B) * coords_B[:, 1:-3]
            + (-2.0 * lamb_M_B - mu_B) * coords_B[:, 2:-2]
            + (lamb_M_B + mu_B) * coords_B[:, 3:-1]
            - mu_2_B * coords_B[:, 4:]
        )
        # boundaries
        a_n_dt_B[:, -2] = (
            -mu_2_B * coords_B[:, -4]
            + (lamb_M_B + mu_B) * coords_B[:, -3]
            + (-lamb_E_B - lamb_M_B - mu_B) * coords_B[:, -2]
            + (lamb_E_B + mu_2_B) * coords_B[:, -1]
        )
        a_n_dt_B[:, -1] = (
            -mu_2_B * coords_B[:, -3]
            + (lamb_E_B + mu_2_B) * coords_B[:, -2]
            - lamb_E_B * coords_B[:, -1]
        )

        a_n_dt_B *= 2.0 * k_B / gamma_B * dt
        a_n_dt_B[:, N_B] += (
            -k_spring / gamma_B * (coords_B[:, N_B] - coords_A[:, N_A]) * dt
        )

        # ===== Corrector =====
        k_n_dt_A[:] = 0.0
        k_n_dt_tmp_A = 0.5 * (a_n_dt_A + dW_A + dH_A)

        k_n_dt_B[:] = 0.0
        k_n_dt_tmp_B = 0.5 * (a_n_dt_B + dW_B + dH_B)

        # A corrector
        k_n_dt_A[:, 0] = (
            -lamb_E_A * k_n_dt_tmp_A[:, 0]
            + (lamb_E_A + mu_2_A) * k_n_dt_tmp_A[:, 1]
            - mu_2_A * k_n_dt_tmp_A[:, 2]
        )
        k_n_dt_A[:, 1] = (
            (lamb_E_A + mu_2_A) * k_n_dt_tmp_A[:, 0]
            + (-lamb_E_A - lamb_M_A - mu_A) * k_n_dt_tmp_A[:, 1]
            + (lamb_M_A + mu_A) * k_n_dt_tmp_A[:, 2]
            - mu_2_A * k_n_dt_tmp_A[:, 3]
        )
        k_n_dt_A[:, 2:-2] = (
            -mu_2_A * k_n_dt_tmp_A[:, :-4]
            + (lamb_M_A + mu_A) * k_n_dt_tmp_A[:, 1:-3]
            + (-2.0 * lamb_M_A - mu_A) * k_n_dt_tmp_A[:, 2:-2]
            + (lamb_M_A + mu_A) * k_n_dt_tmp_A[:, 3:-1]
            - mu_2_A * k_n_dt_tmp_A[:, 4:]
        )
        k_n_dt_A[:, -2] = (
            -mu_2_A * k_n_dt_tmp_A[:, -4]
            + (lamb_M_A + mu_A) * k_n_dt_tmp_A[:, -3]
            + (-lamb_E_A - lamb_M_A - mu_A) * k_n_dt_tmp_A[:, -2]
            + (lamb_E_A + mu_2_A) * k_n_dt_tmp_A[:, -1]
        )
        k_n_dt_A[:, -1] = (
            -mu_2_A * k_n_dt_tmp_A[:, -3]
            + (lamb_E_A + mu_2_A) * k_n_dt_tmp_A[:, -2]
            - lamb_E_A * k_n_dt_tmp_A[:, -1]
        )

        corr_n_dt_A = 2.0 * k_A / gamma_A * k_n_dt_A * dt
        corr_n_dt_A[:, N_A] += (
            -k_spring
            / gamma_A
            * (k_n_dt_tmp_A[:, N_A] - k_n_dt_tmp_B[:, N_B])
            * dt
        )

        coords_A += a_n_dt_A + corr_n_dt_A + dW_A

        # B corrector
        k_n_dt_B[:, 0] = (
            -lamb_E_B * k_n_dt_tmp_B[:, 0]
            + (lamb_E_B + mu_2_B) * k_n_dt_tmp_B[:, 1]
            - mu_2_B * k_n_dt_tmp_B[:, 2]
        )
        k_n_dt_B[:, 1] = (
            (lamb_E_B + mu_2_B) * k_n_dt_tmp_B[:, 0]
            + (-lamb_E_B - lamb_M_B - mu_B) * k_n_dt_tmp_B[:, 1]
            + (lamb_M_B + mu_B) * k_n_dt_tmp_B[:, 2]
            - mu_2_B * k_n_dt_tmp_B[:, 3]
        )
        k_n_dt_B[:, 2:-2] = (
            -mu_2_B * k_n_dt_tmp_B[:, :-4]
            + (lamb_M_B + mu_B) * k_n_dt_tmp_B[:, 1:-3]
            + (-2.0 * lamb_M_B - mu_B) * k_n_dt_tmp_B[:, 2:-2]
            + (lamb_M_B + mu_B) * k_n_dt_tmp_B[:, 3:-1]
            - mu_2_B * k_n_dt_tmp_B[:, 4:]
        )
        k_n_dt_B[:, -2] = (
            -mu_2_B * k_n_dt_tmp_B[:, -4]
            + (lamb_M_B + mu_B) * k_n_dt_tmp_B[:, -3]
            + (-lamb_E_B - lamb_M_B - mu_B) * k_n_dt_tmp_B[:, -2]
            + (lamb_E_B + mu_2_B) * k_n_dt_tmp_B[:, -1]
        )
        k_n_dt_B[:, -1] = (
            -mu_2_B * k_n_dt_tmp_B[:, -3]
            + (lamb_E_B + mu_2_B) * k_n_dt_tmp_B[:, -2]
            - lamb_E_B * k_n_dt_tmp_B[:, -1]
        )

        corr_n_dt_B = 2.0 * k_B / gamma_B * k_n_dt_B * dt
        corr_n_dt_B[:, N_B] += (
            -k_spring
            / gamma_B
            * (k_n_dt_tmp_B[:, N_B] - k_n_dt_tmp_A[:, N_A])
            * dt
        )

        coords_B += a_n_dt_B + corr_n_dt_B + dW_B

    # ------------------------------------------------------------------ #
    # MSD calculation
    # ------------------------------------------------------------------ #
    burning_ratio = 20
    t_A, msd_A = concat_msds_with_mask(
        traj_A_dict, Tperiods=Tperiods, dt=dt, burning_ratio=burning_ratio
    )
    t_B, msd_B = concat_msds_with_mask(
        traj_B_dict, Tperiods=Tperiods, dt=dt, burning_ratio=burning_ratio
    )

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    out_path = os.path.join(
        args.DIR,
        f"{args.prefix}_kA{k_A}_kS{k_spring}.npz",
    )
    np.savez(out_path, tLags=t_A, msd_A=msd_A, msd_B=msd_B)
    print(f"Saved MSD data to: {out_path}")


if __name__ == "__main__":
    main()
