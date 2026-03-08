import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Configuración por defecto
# =========================
DEFAULT_SUCCESS_THRESH_M = 0.04    # 40 mm
DEFAULT_FINAL_WINDOW_S = 0.5       # últimos 0.5 s del dwell
DEFAULT_DWELL_SEC = 1.0            # si no se detecta metadata
DEFAULT_WAYPOINT_TOL = 1e-9        # tolerancia para detectar cambio exacto en p_des


# =========================
# Utilidades básicas
# =========================
def _require_columns(df: pd.DataFrame, cols: List[str], csv_path: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: faltan columnas requeridas: {missing}")


def _find_time_column(df: pd.DataFrame) -> str:
    for c in ["t", "time_rel", "time", "timestamp_rel"]:
        if c in df.columns:
            return c
    raise ValueError("No encontré columna de tiempo relativa. Esperaba una de: t, time_rel, time, timestamp_rel")


def _joint_error_columns(df: pd.DataFrame) -> List[str]:
    cols = [f"e{i}" for i in range(1, 7)]
    _require_columns(df, cols, "<dataframe>")
    return cols


def _task_space_available(df: pd.DataFrame) -> bool:
    needed = ["px", "py", "pz", "px_des", "py_des", "pz_des"]
    return all(c in df.columns for c in needed)


def _compute_ee_error(df: pd.DataFrame) -> np.ndarray:
    _require_columns(df, ["px", "py", "pz", "px_des", "py_des", "pz_des"], "<dataframe>")
    err = np.sqrt(
        (df["px"].to_numpy(dtype=float) - df["px_des"].to_numpy(dtype=float)) ** 2 +
        (df["py"].to_numpy(dtype=float) - df["py_des"].to_numpy(dtype=float)) ** 2 +
        (df["pz"].to_numpy(dtype=float) - df["pz_des"].to_numpy(dtype=float)) ** 2
    )
    return err


# =========================
# Detección de dwell windows
# =========================
def detect_dwell_windows(
    df: pd.DataFrame,
    time_col: str,
    dwell_sec: float = DEFAULT_DWELL_SEC,
    waypoint_tol: float = DEFAULT_WAYPOINT_TOL,
) -> List[Dict]:
    """
    Detecta ventanas donde p_des se mantiene constante.
    Regresa lista de dicts:
      {
        "waypoint_idx": int,
        "start_idx": int,
        "end_idx": int,
        "t0": float,
        "t1": float,
        "duration": float,
        "p_des": (x, y, z)
      }
    """
    if not _task_space_available(df):
        return []

    pdes = df[["px_des", "py_des", "pz_des"]].to_numpy(dtype=float)
    t = df[time_col].to_numpy(dtype=float)

    if len(df) < 2:
        return []

    # Marca cambio cuando el target deseado cambia entre muestras consecutivas
    change = np.any(np.abs(np.diff(pdes, axis=0)) > waypoint_tol, axis=1)

    # Segmentos constantes de p_des
    boundaries = [0]
    boundaries.extend(list(np.where(change)[0] + 1))
    boundaries.append(len(df))

    dwell_windows = []
    wp_counter = 1

    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1] - 1
        if e <= s:
            continue

        duration = float(t[e] - t[s])
        if duration + 1e-12 >= dwell_sec:
            dwell_windows.append({
                "waypoint_idx": wp_counter,
                "start_idx": int(s),
                "end_idx": int(e),
                "t0": float(t[s]),
                "t1": float(t[e]),
                "duration": duration,
                "p_des": tuple(map(float, pdes[s]))
            })
            wp_counter += 1

    return dwell_windows


# =========================
# Métricas principales
# =========================
def compute_metrics(
    csv_path: str,
    success_thresh_m: float = DEFAULT_SUCCESS_THRESH_M,
    final_window_s: float = DEFAULT_FINAL_WINDOW_S,
    dwell_sec: float = DEFAULT_DWELL_SEC,
) -> Dict:
    df = pd.read_csv(csv_path)
    time_col = _find_time_column(df)

    # Joint-space metrics
    e_cols = _joint_error_columns(df)
    E = df[e_cols].to_numpy(dtype=float)

    rmse_joints = np.sqrt(np.mean(E ** 2, axis=0))
    emax_joints = np.max(np.abs(E), axis=0)

    # Task-space metrics
    if _task_space_available(df):
        e_ee = _compute_ee_error(df)
        rmse_ee_m = float(np.sqrt(np.mean(e_ee ** 2)))
        max_ee_m = float(np.max(e_ee))
    else:
        e_ee = np.array([])
        rmse_ee_m = 0.0
        max_ee_m = 0.0

    # Dwell-window metrics
    dwell_windows = detect_dwell_windows(df, time_col=time_col, dwell_sec=dwell_sec)

    dwell_joint_mean_errors = []   # lista por waypoint, cada entrada tiene 6 joints
    dwell_ee_stats = []            # lista por waypoint: mean/max eee
    success_flags = []

    if dwell_windows and _task_space_available(df):
        t = df[time_col].to_numpy(dtype=float)

        for dw in dwell_windows:
            s = dw["start_idx"]
            e = dw["end_idx"] + 1

            E_dw = np.abs(E[s:e, :])
            joint_mean = np.mean(E_dw, axis=0)
            dwell_joint_mean_errors.append(joint_mean)

            ee_dw = e_ee[s:e]
            ee_mean = float(np.mean(ee_dw))
            ee_max = float(np.max(ee_dw))
            dwell_ee_stats.append({
                "waypoint_idx": dw["waypoint_idx"],
                "t0": dw["t0"],
                "t1": dw["t1"],
                "duration": dw["duration"],
                "p_des": dw["p_des"],
                "ee_mean_m": ee_mean,
                "ee_max_m": ee_max,
            })

            # Success usando los últimos final_window_s del dwell
            t1 = dw["t1"]
            mask_final = (t[s:e] >= (t1 - final_window_s))
            ee_final = ee_dw[mask_final] if np.any(mask_final) else ee_dw
            success = bool(np.all(ee_final < success_thresh_m))
            success_flags.append(success)

    elif dwell_windows:
        # Si no hay task-space, al menos calcula dwell de juntas
        for dw in dwell_windows:
            s = dw["start_idx"]
            e = dw["end_idx"] + 1
            E_dw = np.abs(E[s:e, :])
            joint_mean = np.mean(E_dw, axis=0)
            dwell_joint_mean_errors.append(joint_mean)

    # Resúmenes de dwell
    if len(dwell_joint_mean_errors) > 0:
        dwell_joint_mean_errors_arr = np.vstack(dwell_joint_mean_errors)  # shape: [num_wp, 6]
        dwell_joint_mean_per_joint = np.mean(dwell_joint_mean_errors_arr, axis=0)
        dwell_joint_mean_avg = float(np.mean(dwell_joint_mean_per_joint))
    else:
        dwell_joint_mean_errors_arr = np.empty((0, 6))
        dwell_joint_mean_per_joint = np.zeros(6)
        dwell_joint_mean_avg = 0.0

    if len(dwell_ee_stats) > 0:
        dwell_ee_mean_avg_m = float(np.mean([d["ee_mean_m"] for d in dwell_ee_stats]))
        dwell_ee_max_avg_m = float(np.mean([d["ee_max_m"] for d in dwell_ee_stats]))
    else:
        dwell_ee_mean_avg_m = 0.0
        dwell_ee_max_avg_m = 0.0

    if len(success_flags) > 0:
        success_rate = 100.0 * float(np.mean(success_flags))
    else:
        success_rate = 0.0

    result = {
        "file": str(csv_path),
        "samples": int(len(df)),
        "time_column": time_col,

        "joint_rmse_per_joint_rad": rmse_joints.tolist(),
        "joint_rmse_avg_rad": float(np.mean(rmse_joints)),

        "joint_emax_per_joint_rad": emax_joints.tolist(),
        "joint_emax_avg_rad": float(np.mean(emax_joints)),

        "dwell_joint_mean_per_joint_rad": dwell_joint_mean_per_joint.tolist(),
        "dwell_joint_mean_avg_rad": dwell_joint_mean_avg,

        "ee_rmse_m": rmse_ee_m,
        "ee_rmse_mm": rmse_ee_m * 1000.0,
        "ee_max_m": max_ee_m,
        "ee_max_mm": max_ee_m * 1000.0,

        "dwell_ee_mean_avg_m": dwell_ee_mean_avg_m,
        "dwell_ee_mean_avg_mm": dwell_ee_mean_avg_m * 1000.0,
        "dwell_ee_max_avg_m": dwell_ee_max_avg_m,
        "dwell_ee_max_avg_mm": dwell_ee_max_avg_m * 1000.0,

        "success_threshold_m": float(success_thresh_m),
        "success_final_window_s": float(final_window_s),
        "waypoint_success_rate_percent": success_rate,

        "num_dwell_windows": int(len(dwell_windows)),
        "dwell_windows": dwell_windows,
        "dwell_ee_per_waypoint": dwell_ee_stats,
        "success_flags_per_waypoint": success_flags,
    }

    return result


# =========================
# Reporte legible
# =========================
def print_report(metrics: Dict) -> None:
    print(f"\n=== MÉTRICAS: {metrics['file']} ===")
    print(f"Muestras:                {metrics['samples']}")
    print(f"Columna de tiempo:       {metrics['time_column']}")
    print(f"Dwell windows detectados:{metrics['num_dwell_windows']}")
    print("-" * 50)

    print("Joint-space")
    print(f"  Joint RMSE avg:        {metrics['joint_rmse_avg_rad']:.6f} rad")
    print(f"  Joint Max avg:         {metrics['joint_emax_avg_rad']:.6f} rad")
    print(f"  Dwell mean err avg:    {metrics['dwell_joint_mean_avg_rad']:.6f} rad")
    print("  RMSE por junta [rad]:  " + ", ".join(f"{x:.6f}" for x in metrics["joint_rmse_per_joint_rad"]))
    print("  Emax por junta [rad]:  " + ", ".join(f"{x:.6f}" for x in metrics["joint_emax_per_joint_rad"]))

    print("-" * 50)
    print("Task-space")
    print(f"  EE RMSE:               {metrics['ee_rmse_mm']:.2f} mm")
    print(f"  EE Max Error:          {metrics['ee_max_mm']:.2f} mm")
    print(f"  Dwell EE mean avg:     {metrics['dwell_ee_mean_avg_mm']:.2f} mm")
    print(f"  Dwell EE max avg:      {metrics['dwell_ee_max_avg_mm']:.2f} mm")

    print("-" * 50)
    print("Waypoint success")
    print(f"  Threshold:             {metrics['success_threshold_m']*1000.0:.2f} mm")
    print(f"  Final window:          {metrics['success_final_window_s']:.2f} s")
    print(f"  Success rate:          {metrics['waypoint_success_rate_percent']:.1f} %")

    if metrics["success_flags_per_waypoint"]:
        flags = ["OK" if s else "FAIL" for s in metrics["success_flags_per_waypoint"]]
        print(f"  Per-waypoint:          {flags}")


# =========================
# Export helpers
# =========================
def flatten_summary_row(metrics: Dict) -> Dict:
    row = {
        "file": metrics["file"],
        "samples": metrics["samples"],
        "num_dwell_windows": metrics["num_dwell_windows"],

        "joint_rmse_avg_rad": metrics["joint_rmse_avg_rad"],
        "joint_emax_avg_rad": metrics["joint_emax_avg_rad"],
        "dwell_joint_mean_avg_rad": metrics["dwell_joint_mean_avg_rad"],

        "ee_rmse_mm": metrics["ee_rmse_mm"],
        "ee_max_mm": metrics["ee_max_mm"],
        "dwell_ee_mean_avg_mm": metrics["dwell_ee_mean_avg_mm"],
        "dwell_ee_max_avg_mm": metrics["dwell_ee_max_avg_mm"],

        "success_threshold_mm": metrics["success_threshold_m"] * 1000.0,
        "success_final_window_s": metrics["success_final_window_s"],
        "waypoint_success_rate_percent": metrics["waypoint_success_rate_percent"],
    }

    for i, val in enumerate(metrics["joint_rmse_per_joint_rad"], start=1):
        row[f"joint{i}_rmse_rad"] = val
    for i, val in enumerate(metrics["joint_emax_per_joint_rad"], start=1):
        row[f"joint{i}_emax_rad"] = val
    for i, val in enumerate(metrics["dwell_joint_mean_per_joint_rad"], start=1):
        row[f"joint{i}_dwell_mean_rad"] = val

    return row


def save_metrics_json(metrics: Dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_summary_csv(rows: List[Dict], out_path: str) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Compute TE300XB metrics from CSV logs.")
    parser.add_argument("csvs", nargs="+", help="Uno o más archivos CSV")
    parser.add_argument("--success-thresh-mm", type=float, default=40.0,
                        help="Threshold de éxito en mm (default: 40.0)")
    parser.add_argument("--final-window", type=float, default=0.5,
                        help="Ventana final del dwell para success rate, en segundos (default: 0.5)")
    parser.add_argument("--dwell-sec", type=float, default=1.0,
                        help="Duración mínima para considerar un segmento como dwell (default: 1.0)")
    parser.add_argument("--save-json", action="store_true",
                        help="Guardar un JSON por cada CSV")
    parser.add_argument("--summary-csv", type=str, default="summary_metrics.csv",
                        help="Nombre del CSV resumen cuando se procesan varios archivos")
    return parser.parse_args()


def main():
    args = parse_args()

    success_thresh_m = args.success_thresh_mm / 1000.0
    all_rows = []

    for csv_path in args.csvs:
        try:
            metrics = compute_metrics(
                csv_path,
                success_thresh_m=success_thresh_m,
                final_window_s=args.final_window,
                dwell_sec=args.dwell_sec,
            )
            print_report(metrics)
            all_rows.append(flatten_summary_row(metrics))

            if args.save_json:
                out_json = str(Path(csv_path).with_suffix(".metrics.json"))
                save_metrics_json(metrics, out_json)
                print(f"JSON guardado en: {out_json}")

        except Exception as e:
            print(f"Error procesando {csv_path}: {e}")

    if all_rows:
        save_summary_csv(all_rows, args.summary_csv)
        print(f"\nResumen CSV guardado en: {args.summary_csv}")


if __name__ == "__main__":
    main()