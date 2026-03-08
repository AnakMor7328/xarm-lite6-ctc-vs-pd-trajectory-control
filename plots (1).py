import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SUCCESS_THRESH_MM = 40.0


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    task_cols = ['px', 'py', 'pz', 'px_des', 'py_des', 'pz_des']
    if all(col in df.columns for col in task_cols):
        df['ee_error_norm'] = np.sqrt(
            (df['px'] - df['px_des'])**2 +
            (df['py'] - df['py_des'])**2 +
            (df['pz'] - df['pz_des'])**2
        )
    else:
        df['ee_error_norm'] = np.nan

    return df


def shade_perturbation(ax, df):
    if 'pert_flag' not in df.columns:
        return

    t = df['t'].to_numpy()
    pert = df['pert_flag'].fillna(0).to_numpy(dtype=int)

    in_region = False
    start_t = None

    for i in range(len(pert)):
        if pert[i] == 1 and not in_region:
            in_region = True
            start_t = t[i]
        elif pert[i] == 0 and in_region:
            in_region = False
            ax.axvspan(start_t, t[i], alpha=0.15, color='gray')

    if in_region:
        ax.axvspan(start_t, t[-1], alpha=0.15, color='gray')


def plot_joint_tracking(df, base_path):
    required = [f'q{i}' for i in range(1, 7)] + [f'qdes{i}' for i in range(1, 7)] + ['t']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Omitiendo seguimiento de juntas para {base_path}: faltan columnas {missing}")
        return

    t = df['t'].to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()

    for j in range(6):
        axes[j].plot(t, df[f'q{j+1}'], label='Medido', alpha=0.8)
        axes[j].plot(t, df[f'qdes{j+1}'], '--', label='Deseado', alpha=0.8)
        axes[j].set_title(f"Seguimiento Junta {j+1}")
        axes[j].set_xlabel("Tiempo [s]")
        axes[j].set_ylabel("Posición [rad]")
        axes[j].grid(True, which='both', linestyle='--', alpha=0.5)
        shade_perturbation(axes[j], df)
        axes[j].legend()

    fig.suptitle(f"Seguimiento de Juntas - Archivo: {base_path}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{base_path}_joint_tracking.png", dpi=200)
    plt.close(fig)


def plot_joint_errors(df, base_path):
    required_q = [f'q{i}' for i in range(1, 7)] + [f'qdes{i}' for i in range(1, 7)] + ['t']
    has_e = all(f'e{i}' in df.columns for i in range(1, 7))
    if not has_e:
        missing = [c for c in required_q if c not in df.columns]
        if missing:
            print(f"Omitiendo errores de juntas para {base_path}: faltan columnas {missing}")
            return

    t = df['t'].to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()

    for j in range(6):
        error = df[f'e{j+1}'] if f'e{j+1}' in df.columns else (df[f'qdes{j+1}'] - df[f'q{j+1}'])
        axes[j].plot(t, error, label='Error')
        axes[j].axhline(0.0, linestyle=':', alpha=0.6)
        axes[j].set_title(f"Error Junta {j+1}")
        axes[j].set_xlabel("Tiempo [s]")
        axes[j].set_ylabel("Error [rad]")
        axes[j].grid(True, which='both', linestyle='--', alpha=0.5)
        shade_perturbation(axes[j], df)

    fig.suptitle(f"Errores de Juntas - Archivo: {base_path}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{base_path}_joint_errors.png", dpi=200)
    plt.close(fig)


def plot_task_space(df, base_path, thr_mm=SUCCESS_THRESH_MM):
    if 'px' not in df.columns or 't' not in df.columns:
        print(f"Omitiendo gráficas de espacio de tarea para {base_path}: faltan datos XYZ o tiempo.")
        return

    t = df['t'].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for i, axis_name in enumerate(['x', 'y', 'z']):
        real_col = f'p{axis_name}'
        des_col = f'p{axis_name}_des'
        if real_col not in df.columns or des_col not in df.columns:
            print(f"Omitiendo eje {axis_name.upper()} para {base_path}: faltan {real_col} o {des_col}")
            continue
        axes[i].plot(t, df[real_col], label=f'Real {axis_name.upper()}')
        axes[i].plot(t, df[des_col], '--', label=f'Deseado {axis_name.upper()}')
        axes[i].set_ylabel(f"{axis_name.upper()} [m]")
        axes[i].grid(True)
        shade_perturbation(axes[i], df)
        axes[i].legend()

    axes[-1].set_xlabel("Tiempo [s]")
    fig.suptitle(f"Seguimiento Espacio de Tarea - Archivo: {base_path}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{base_path}_task_xyz.png", dpi=200)
    plt.close(fig)

    needed_3d = ['px', 'py', 'pz', 'px_des', 'py_des', 'pz_des']
    if all(c in df.columns for c in needed_3d):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(df['px'], df['py'], df['pz'], label='Real', lw=2)
        ax.plot(df['px_des'], df['py_des'], df['pz_des'], '--', label='Deseado', alpha=0.7)
        ax.scatter(df['px_des'].iloc[0], df['py_des'].iloc[0], df['pz_des'].iloc[0], s=50, label='Inicio')
        ax.scatter(df['px_des'].iloc[-1], df['py_des'].iloc[-1], df['pz_des'].iloc[-1], s=50, label='Final')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f"Trayectoria 3D - {base_path}")
        ax.legend()
        plt.savefig(f"{base_path}_3d_path.png", dpi=200)
        plt.close(fig)

    if 'ee_error_norm' in df.columns and not df['ee_error_norm'].isnull().all():
        plt.figure(figsize=(12, 5))
        plt.plot(t, df['ee_error_norm'] * 1000, label='Error EE')
        plt.axhline(thr_mm, linestyle='--', label=f'Umbral de éxito ({thr_mm:.1f} mm)')
        shade_perturbation(plt.gca(), df)
        plt.title(f"Norma del Error del Efector Final - {base_path}")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Error [mm]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{base_path}_ee_error_norm.png", dpi=200)
        plt.close()


def plot_phase_portraits(df, base_path, joints=(1, 2, 4), stride=10):
    if 't' not in df.columns:
        print(f"Omitiendo retratos de fase para {base_path}: falta columna t")
        return

    t = df['t'].to_numpy()

    for j in joints:
        if f'e{j}' in df.columns:
            error = df[f'e{j}']
        else:
            q_col = f'q{j}'
            qdes_col = f'qdes{j}'
            if q_col not in df.columns or qdes_col not in df.columns:
                print(f"Omitiendo retrato de fase junta {j} para {base_path}: faltan {q_col} o {qdes_col}")
                continue
            error = df[qdes_col] - df[q_col]

        edot = np.gradient(error, t)

        err_ds = error[::stride]
        edot_ds = edot[::stride]

        ex1, ex2 = np.percentile(err_ds, [1, 99])
        ey1, ey2 = np.percentile(edot_ds, [1, 99])

        plt.figure(figsize=(7, 7))
        plt.scatter(err_ds, edot_ds, s=6, alpha=0.25)
        plt.scatter([0], [0], s=60, label='Equilibrio')
        plt.xlim(ex1, ex2)
        plt.ylim(ey1, ey2)
        plt.xlabel(f"Error Junta {j} [rad]")
        plt.ylabel(f"Derivada del Error [rad/s]")
        plt.title(f"Retrato de Fase Junta {j} - {base_path}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{base_path}_phase_j{j}.png", dpi=200)
        plt.close()


def run_analysis(csv_path, thr_mm=SUCCESS_THRESH_MM):
    print(f"Analizando archivo: {csv_path}")
    try:
        df = load_data(csv_path)
        base = csv_path.replace(".csv", "")

        plot_joint_tracking(df, base)
        plot_joint_errors(df, base)
        plot_task_space(df, base, thr_mm=thr_mm)
        plot_phase_portraits(df, base)

        print(f"¡Hecho! Imágenes guardadas para {base}")
    except Exception as e:
        print(f"Error al procesar {csv_path}: {e}")


def overlay_task_xyz(csv_pd, csv_ctc, out_prefix="compare_overlay"):
    dfp = load_data(csv_pd)
    dfc = load_data(csv_ctc)

    needed = ['t', 'px', 'py', 'pz', 'px_des', 'py_des', 'pz_des']
    if not all(c in dfp.columns for c in needed) or not all(c in dfc.columns for c in needed):
        print("No hay columnas completas px/py/pz y deseados en ambos CSV. No puedo hacer overlay task-space.")
        return

    t1 = dfp['t'].to_numpy()
    t2 = dfc['t'].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(t1, dfp['px'], label='PD real X')
    axes[0].plot(t2, dfc['px'], label='CTC real X', linestyle='--')
    axes[0].plot(t1, dfp['px_des'], label='Deseado X', linestyle=':')
    axes[0].set_ylabel("X [m]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t1, dfp['py'], label='PD real Y')
    axes[1].plot(t2, dfc['py'], label='CTC real Y', linestyle='--')
    axes[1].plot(t1, dfp['py_des'], label='Deseado Y', linestyle=':')
    axes[1].set_ylabel("Y [m]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t1, dfp['pz'], label='PD real Z')
    axes[2].plot(t2, dfc['pz'], label='CTC real Z', linestyle='--')
    axes[2].plot(t1, dfp['pz_des'], label='Deseado Z', linestyle=':')
    axes[2].set_ylabel("Z [m]")
    axes[2].set_xlabel("t [s]")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle("Task-space XYZ comparison (PD vs CTC)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{out_prefix}_task_xyz_overlay.png", dpi=200)
    plt.close(fig)


def overlay_ee_error(csv_pd, csv_ctc, out_prefix="compare_overlay", thr_mm=SUCCESS_THRESH_MM):
    dfp = load_data(csv_pd)
    dfc = load_data(csv_ctc)

    if dfp['ee_error_norm'].isnull().all() or dfc['ee_error_norm'].isnull().all():
        print("No hay ee_error_norm. Revisa columnas px/py/pz y *_des.")
        return

    t1 = dfp['t'].to_numpy()
    t2 = dfc['t'].to_numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(t1, dfp['ee_error_norm'] * 1000, label='PD EE error')
    plt.plot(t2, dfc['ee_error_norm'] * 1000, label='CTC EE error', linestyle='--')
    plt.axhline(thr_mm, linestyle=':', label=f'Umbral {thr_mm:.1f} mm')
    plt.title("EE error norm comparison (PD vs CTC)")
    plt.xlabel("t [s]")
    plt.ylabel("Error [mm]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_ee_error_overlay.png", dpi=200)
    plt.close()


def infer_compare_prefix(csv_a, csv_b):
    a = csv_a.lower()
    b = csv_b.lower()

    # primero revisar no_perturb
    if "no_perturb" in a or "no_perturb" in b:
        return "compare_no_perturb"

    # luego perturbado
    if "trial_" in a or "trial_" in b or "_pert" in a or "_pert" in b:
        return "compare_perturb"

    return "compare_overlay"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera gráficas individuales o comparativas para PD/CTC."
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("PD_CSV", "CTC_CSV"),
        help="Genera comparación entre dos CSV"
    )
    parser.add_argument(
        "--thr-mm",
        type=float,
        default=SUCCESS_THRESH_MM,
        help=f"Umbral de éxito en mm (default: {SUCCESS_THRESH_MM})"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Uno o más CSV para análisis individual"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.compare is not None:
        pd_csv, ctc_csv = args.compare
        out_prefix = infer_compare_prefix(pd_csv, ctc_csv)

        overlay_task_xyz(pd_csv, ctc_csv, out_prefix=out_prefix)
        overlay_ee_error(pd_csv, ctc_csv, out_prefix=out_prefix, thr_mm=args.thr_mm)

        print(f"¡Hecho! Imágenes comparativas guardadas con prefijo: {out_prefix}")
    elif args.inputs:
        for path in args.inputs:
            run_analysis(path, thr_mm=args.thr_mm)
    else:
        print("Uso:")
        print("  python3 plots.py archivo.csv")
        print("  python3 plots.py archivo1.csv archivo2.csv")
        print("  python3 plots.py --compare pd.csv ctc.csv")