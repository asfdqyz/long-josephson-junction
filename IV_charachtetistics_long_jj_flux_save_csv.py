from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def system(t, y, N, alpha, i, l, a):
    """
    Open chain, nodes k=0..N-1, cells m=0..N-2 between (m) and (m+1).
    a[m] = 2π * Φ_e(m) / Φ0  (external flux in cell m)

    Equations (normalized):
      dphi/dt = psi
      dpsi/dt = i - sin(phi) - alpha*psi - coupling

    coupling:
      left:   (phi0 - phi1 + a0)/l
      bulk:   (2phi_k - phi_{k-1} - phi_{k+1} + a_k - a_{k-1})/l
      right:  (phi_{N-1} - phi_{N-2} - a_{N-2})/l
    """
    phi = y[:N]
    psi = y[N:]

    a = np.zeros(N - 1, dtype=float) if a is None else np.asarray(a, dtype=float)
    if a.ndim == 0:
        a = np.full(N - 1, float(a))
    if a.shape[0] != N - 1:
        raise ValueError(f"a must have length N-1={N-1}, got {a.shape[0]}")

    dphi_dt = psi
    dpsi_dt = np.zeros_like(psi)

    # left boundary k=0
    coupling0 = (phi[0] - phi[1] + a[0]) / l
    dpsi_dt[0] = i - np.sin(phi[0]) - alpha * psi[0] - coupling0

    # interior k=1..N-2
    if N > 2:
        coupling_mid = (2.0 * phi[1:-1] - phi[0:-2] - phi[2:] + a[1:] - a[:-1]) / l
        dpsi_dt[1:-1] = i - np.sin(phi[1:-1]) - alpha * psi[1:-1] - coupling_mid

    # right boundary k=N-1
    couplingN = (phi[-1] - phi[-2] - a[-1]) / l
    dpsi_dt[-1] = i - np.sin(phi[-1]) - alpha * psi[-1] - couplingN

    return np.concatenate([dphi_dt, dpsi_dt])


def avg_voltage_single_node(sol, N, transient_fraction, node_index):
    cut = int(transient_fraction * sol.t.size)
    psi = sol.y[N:, cut:]  # (N, T-cut)
    return float(np.mean(psi[node_index]))


def simulate_IV_up_single_node(
    N: int,
    alpha: float,
    l: float,
    y0: np.ndarray,
    i_values: np.ndarray,
    node_index: int,
    a=None,
    t_end: float = 1500.0,
    n_eval: int = 6000,
    method: str = "RK45",
    transient_fraction: float = 0.8,
    show_progress: bool = True,
):
    if node_index < 0 or node_index >= N:
        raise ValueError(f"node_index must be in [0, {N-1}], got {node_index}")

    t_span = (0.0, t_end)
    t_eval = np.linspace(*t_span, n_eval)

    def progress_line(k: int, total: int, i_val: float, t0: float):
        if not show_progress:
            return
        done = k + 1
        frac = done / total
        bar_len = 28
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - t0
        rate = elapsed / done
        eta = rate * (total - done)
        print(
            f"\r[{bar}] {done:>4}/{total:<4}  i={i_val: .4f}  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s",
            end="",
            flush=True,
        )

    y_init = y0.copy()
    V = []
    t0 = time.time()
    total = len(i_values)

    for k, i_val in enumerate(i_values):
        sol = solve_ivp(
            system,
            t_span,
            y_init,
            args=(N, alpha, float(i_val), l, a),
            t_eval=t_eval,
            method=method,
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed at i={i_val}: {sol.message}")

        # continuation
        y_init = sol.y[:, -1].copy()
        V.append(avg_voltage_single_node(sol, N, transient_fraction, node_index))
        progress_line(k, total, float(i_val), t0)

    if show_progress:
        print()

    return np.array(V)


def run_one_flux(job):
    (f, N, alpha, l, y0, currents, node_index, t_end, n_eval, method, transient_fraction) = job
    a = 2.0 * np.pi * f * np.ones(N - 1)


    V = simulate_IV_up_single_node(
        N=N,
        alpha=alpha,
        l=l,
        y0=y0,
        i_values=currents,
        node_index=node_index,
        a=a,
        t_end=t_end,
        n_eval=n_eval,
        method=method,
        transient_fraction=transient_fraction,
        show_progress=False,  
    )
    return float(f), V


def fmt_float_for_filename(x, nd=6):
    """Make a float safe for filenames: 0.01 -> 0p01, -1.2 -> m1p2."""
    s = f"{float(x):.{nd}g}"
    return s.replace("-", "m").replace(".", "p")


def save_single_flux_csv(output_dir, f, currents, V):
    """Save one I-V curve for one flux value."""
    filename = f"IV_flux_{fmt_float_for_filename(f)}Phi0.csv"
    path = os.path.join(output_dir, filename)

    data = np.column_stack([currents, V])
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="I_normalized,V_normalized",
        comments="",
    )
    return path


def save_all_fluxes_csv(output_dir, currents, results):
    """Save all I-V curves into one wide CSV table."""
    f_sorted = sorted(results.keys())
    columns = [currents] + [results[f] for f in f_sorted]

    header = "I_normalized," + ",".join(
        f"V_normalized_flux_{fmt_float_for_filename(f)}Phi0" for f in f_sorted
    )

    path = os.path.join(output_dir, "IV_all_fluxes.csv")
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=header,
        comments="",
    )
    return path


def save_parameters_csv(output_dir, params):
    """Save calculation parameters into a separate CSV file."""
    path = os.path.join(output_dir, "parameters.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("parameter,value\n")
        for key, value in params.items():
            f.write(f"{key},{value}\n")
    return path



if __name__ == "__main__":
    # ---- parameters ----
    N = 100
    alpha = 0.05
    l = 0.01

    phi0 = np.zeros(N)
    psi0 = np.zeros(N)
    y0 = np.concatenate([phi0, psi0])

    node_index = 1  

    i_max = 1.1
    di = 0.01
    currents = np.arange(0.0, i_max + 1e-12, di)

    flux_values = np.arange(8, 8.5 + 1e-12, 0.05 ) 
    t_end = 1500.0
    n_eval = 10000
    method = "RK45"        
    transient_fraction = 0.8

  
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        f"IV_results_N{N}_alpha{fmt_float_for_filename(alpha)}_"
        f"l{fmt_float_for_filename(l)}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"CSV results will be saved to: {output_dir}")

    save_parameters_csv(
        output_dir,
        {
            "N": N,
            "alpha": alpha,
            "l": l,
            "node_index_python": node_index,
            "node_index_human": node_index + 1,
            "i_max": i_max,
            "di": di,
            "flux_values": ";".join(str(float(f)) for f in flux_values),
            "t_end": t_end,
            "n_eval": n_eval,
            "method": method,
            "transient_fraction": transient_fraction,
        },
    )

    cpu = os.cpu_count() or 2
    max_workers = min(len(flux_values), max(1, cpu - 1))

    jobs = [
        (float(f), N, alpha, l, y0, currents, node_index, t_end, n_eval, method, transient_fraction)
        for f in flux_values
    ]

    results = {}  

    t0 = time.time()
    print(f"Running {len(jobs)} flux values in parallel with max_workers={max_workers} ...")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one_flux, job) for job in jobs]
        done_count = 0

        for fut in as_completed(futures):
            f, V = fut.result()
            results[f] = V

            saved_path = save_single_flux_csv(output_dir, f, currents, V)

            done_count += 1
            elapsed = time.time() - t0
            print(
                f"Done {done_count}/{len(jobs)}: f={f:.2f}   elapsed={elapsed:.1f}s   saved={saved_path}"
            )

    all_path = save_all_fluxes_csv(output_dir, currents, results)
    print(f"Saved combined CSV: {all_path}")

    def fmt_f(f, nd=3):
        s = f"{float(f):.{nd}f}".rstrip("0").rstrip(".")
        return s

    plt.figure(figsize=(8, 6))

    f_sorted = sorted(results.keys())
    cmap = plt.get_cmap("tab20")  
    colors = cmap(np.linspace(0.0, 1.0, max(len(f_sorted), 2)))

    for j, f in enumerate(f_sorted):
        label = rf"$\Phi_e = {fmt_f(f)}\,\Phi_0$"
        plt.plot(
            currents,
            results[f],
            "-",
            lw=1.0,           
            color=colors[j],    
            label=label,
        )

    plt.xlabel("i (normalized)")
    plt.ylabel(f"<dphi/dt> at node {node_index+1} (normalized voltage)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
