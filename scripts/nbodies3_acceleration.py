#!/usr/bin/env python3
"""
Mesure l'acceleration de la version MPI2 distribuant le calcul sur plusieurs processus.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VENV_SITE_PACKAGES = sorted((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))
ENVIRONMENT_CSV = REPO_ROOT / "results" / "experiment_environment.csv"
DEFAULT_RESULTS_CSV = REPO_ROOT / "results" / "nbodies3_acceleration.csv"
DEFAULT_TIME_FIGURE = REPO_ROOT / "docs" / "examen" / "imgs" / "nbodies3_time_heatmap.png"
DEFAULT_SPEEDUP_FIGURE = REPO_ROOT / "docs" / "examen" / "imgs" / "nbodies3_speedup_heatmap.png"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mesure le temps de calcul et l'acceleration de nbodies_grid_numba3."
    )
    parser.add_argument(
        "--environment-csv",
        type=Path,
        default=ENVIRONMENT_CSV,
        help="CSV contenant l'environnement d'execution et les parametres du profilage.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="CSV de sortie pour les mesures d'acceleration.",
    )
    parser.add_argument(
        "--output-time-figure",
        type=Path,
        default=DEFAULT_TIME_FIGURE,
        help="Figure PNG de sortie pour le temps moyen.",
    )
    parser.add_argument(
        "--output-speedup-figure",
        type=Path,
        default=DEFAULT_SPEEDUP_FIGURE,
        help="Figure PNG de sortie pour l'acceleration.",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=None,
        help="Nombre maximal de processus MPI a tester. Par defaut, valeur lue depuis experiment_environment.csv.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Nombre maximal de threads numba a tester. Par defaut, valeur lue depuis experiment_environment.csv.",
    )
    parser.add_argument(
        "--threads-list",
        nargs="+",
        type=int,
        default=None,
        help="Liste explicite des nombres de threads Numba a tester.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Mode interne pour executer toutes les mesures d'un nombre fixe de processus MPI.",
    )
    parser.add_argument("--dataset", type=str, help="Jeu de donnees pour le mode worker.")
    parser.add_argument("--dt", type=float, help="Pas de temps pour le mode worker.")
    parser.add_argument("--frames", type=int, help="Nombre d'iterations mesurees pour le mode worker.")
    parser.add_argument("--warmup", type=int, help="Nombre d'iterations d'echauffement pour le mode worker.")
    parser.add_argument("--max-threads-worker", type=int, help="Nombre maximal de threads testes dans le mode worker.")
    parser.add_argument(
        "--threads-list-worker",
        nargs="+",
        type=int,
        default=None,
        help="Liste explicite des threads testes dans le mode worker.",
    )
    parser.add_argument(
        "--cells",
        nargs=3,
        type=int,
        metavar=("NI", "NJ", "NK"),
        help="Nombre de cellules de la grille pour le mode worker.",
    )
    return parser.parse_args()


def read_environment_csv(path: Path) -> dict[tuple[str, str], str]:
    rows: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[(row["categorie"], row["parametre"])] = row["valeur"]
    return rows


def parse_cells(raw_value: str) -> tuple[int, int, int]:
    cleaned = raw_value.strip().strip("()")
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Format de grille invalide: {raw_value}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def extend_with_venv_site_packages() -> None:
    for path in VENV_SITE_PACKAGES:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def worker_measure(
    dataset: str,
    dt: float,
    cells: tuple[int, int, int],
    frames: int,
    warmup: int,
    threads_list: list[int],
) -> list[dict[str, float]] | None:
    extend_with_venv_site_packages()

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    if "visualizer3d" not in sys.modules:
        dummy_visualizer = types.ModuleType("visualizer3d")
        dummy_visualizer.Visualizer3D = object
        sys.modules["visualizer3d"] = dummy_visualizer

    from mpi4py import MPI
    from numba import set_num_threads

    import nbodies_grid_numba3

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows: list[dict[str, float]] = []

    for threads in threads_list:
        set_num_threads(threads)
        local_system = nbodies_grid_numba3.NBodySystem(dataset, ncells_per_dir=cells)

        for _ in range(warmup):
            comm.Barrier()
            nbodies_grid_numba3.distributed_update_positions(local_system, dt, comm)
            comm.Barrier()

        iteration_times = []
        for _ in range(frames):
            comm.Barrier()
            start = perf_counter()
            nbodies_grid_numba3.distributed_update_positions(local_system, dt, comm)
            comm.Barrier()
            end = perf_counter()
            if rank == 0:
                iteration_times.append((end - start) * 1000.0)

        if rank == 0:
            measures = np.array(iteration_times, dtype=np.float64)
            rows.append(
                {
                    "processes": size,
                    "threads": threads,
                    "frames": int(frames),
                    "warmup": int(warmup),
                    "mean_iteration_ms": float(np.mean(measures)),
                    "std_iteration_ms": float(np.std(measures)),
                    "total_measured_ms": float(np.sum(measures)),
                }
            )

    if rank == 0:
        return rows
    return None


def interpreter_with_mpi() -> str:
    for candidate in (sys.executable, "python3"):
        try:
            completed = subprocess.run(
                [candidate, "-c", "import mpi4py"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        if completed.returncode == 0:
            return candidate
    raise RuntimeError("Aucun interpreteur Python avec mpi4py n'a ete trouve.")


def run_worker_subprocess(
    processes: int,
    dataset: str,
    dt: float,
    cells: tuple[int, int, int],
    frames: int,
    warmup: int,
    threads_list: list[int],
) -> list[dict[str, float]]:
    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = str(max(threads_list))
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    python_cmd = interpreter_with_mpi()

    if VENV_SITE_PACKAGES:
        existing_pythonpath = env.get("PYTHONPATH", "")
        venv_paths = os.pathsep.join(str(path) for path in VENV_SITE_PACKAGES)
        env["PYTHONPATH"] = (
            f"{venv_paths}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else venv_paths
        )

    cmd = [
        "mpiexec",
        "-n",
        str(processes),
        python_cmd,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        dataset,
        "--dt",
        str(dt),
        "--frames",
        str(frames),
        "--warmup",
        str(warmup),
        "--threads-list-worker",
        *[str(value) for value in threads_list],
        "--cells",
        str(cells[0]),
        str(cells[1]),
        str(cells[2]),
    ]

    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )

    cleaned_stdout = ANSI_ESCAPE_RE.sub("", completed.stdout)
    output_lines = [line.strip() for line in cleaned_stdout.splitlines() if line.strip()]
    json_line = next((line for line in reversed(output_lines) if line.startswith("[") and line.endswith("]")), "")
    return json.loads(json_line)


def save_results_csv(
    output_csv: Path,
    rows: list[dict[str, float]],
    dataset: str,
    dt: float,
    cells: tuple[int, int, int],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    baseline = next(
        row["mean_iteration_ms"]
        for row in rows
        if int(row["processes"]) == 1 and int(row["threads"]) == 1
    )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "processes",
                "threads",
                "dataset",
                "dt",
                "cells",
                "mean_iteration_ms",
                "std_iteration_ms",
                "total_measured_ms",
                "speedup",
                "efficiency",
            ]
        )
        for row in rows:
            speedup = baseline / row["mean_iteration_ms"]
            efficiency = speedup / (row["processes"] * row["threads"])
            writer.writerow(
                [
                    row["processes"],
                    row["threads"],
                    dataset,
                    dt,
                    str(cells),
                    f"{row['mean_iteration_ms']:.6f}",
                    f"{row['std_iteration_ms']:.6f}",
                    f"{row['total_measured_ms']:.6f}",
                    f"{speedup:.6f}",
                    f"{efficiency:.6f}",
                ]
            )


def build_metric_matrices(
    rows: list[dict[str, float]],
    process_values: list[int],
    thread_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    time_matrix = np.zeros((len(process_values), len(thread_values)), dtype=np.float64)
    speedup_matrix = np.zeros((len(process_values), len(thread_values)), dtype=np.float64)
    baseline = next(
        row["mean_iteration_ms"]
        for row in rows
        if int(row["processes"]) == 1 and int(row["threads"]) == 1
    )
    process_index = {value: index for index, value in enumerate(process_values)}
    thread_index = {value: index for index, value in enumerate(thread_values)}

    for row in rows:
        p = process_index[int(row["processes"])]
        t = thread_index[int(row["threads"])]
        time_matrix[p, t] = row["mean_iteration_ms"]
        speedup_matrix[p, t] = baseline / row["mean_iteration_ms"]

    return time_matrix, speedup_matrix


def create_heatmap(
    output_figure: Path,
    matrix: np.ndarray,
    process_values: list[int],
    thread_values: list[int],
    title: str,
    colorbar_label: str,
    fmt: str,
) -> None:
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    n_processes, n_threads = matrix.shape

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")

    ax.set_title(title)
    ax.set_xlabel("Nombre de threads Numba")
    ax.set_ylabel("Nombre de processus MPI")
    ax.set_xticks(np.arange(n_threads))
    ax.set_xticklabels(thread_values)
    ax.set_yticks(np.arange(n_processes))
    ax.set_yticklabels(process_values)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    for i in range(n_processes):
        for j in range(n_threads):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white", fontsize=8)

    fig.savefig(output_figure, dpi=180)
    plt.close(fig)


def latex_speedup_table(rows: list[dict[str, float]], process_values: list[int], thread_values: list[int]) -> str:
    _, speedup_matrix = build_metric_matrices(rows, process_values, thread_values)
    header = "Processus / Threads & " + " & ".join(str(t) for t in thread_values) + " \\\\"
    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        "    \\small",
        "    \\begin{tabular}{" + "c" * (len(thread_values) + 1) + "}",
        "        \\toprule",
        f"        {header}",
        "        \\midrule",
    ]

    for process_index, process_value in enumerate(process_values):
        values = " & ".join(
            f"{speedup_matrix[process_index, thread_index]:.2f}" for thread_index in range(len(thread_values))
        )
        lines.append(f"        {process_value} & {values} \\\\")

    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    \\caption{Accélération mesurée pour \\texttt{nbodies\\_grid\\_numba3.py} en fonction du nombre de processus MPI et du nombre de threads Numba.}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def update_report(
    report_path: Path,
    time_figure: Path,
    speedup_figure: Path,
    rows: list[dict[str, float]],
    process_values: list[int],
    thread_values: list[int],
) -> None:
    baseline = next(
        row["mean_iteration_ms"]
        for row in rows
        if int(row["processes"]) == 1 and int(row["threads"]) == 1
    )
    best_row = min(rows, key=lambda row: row["mean_iteration_ms"])
    best_speedup = baseline / best_row["mean_iteration_ms"]
    time_figure_rel = time_figure.relative_to(report_path.parent).as_posix()
    speedup_figure_rel = speedup_figure.relative_to(report_path.parent).as_posix()
    results_table = latex_speedup_table(rows, process_values, thread_values)
    process_text = f"de {process_values[0]} à {process_values[-1]}"
    threads_text = ", ".join(str(value) for value in thread_values)

    subsection_text = f"""\\subsection{{Accélération obtenue}}

L'accélération de la version MPI2 a été mesurée à l'aide du script \\texttt{{nbodies3\\_acceleration.py}}. Contrairement à MPI1, cette version distribue le calcul entre plusieurs processus MPI, tout en conservant la parallélisation locale par \\texttt{{numba}} à l'intérieur de chaque processus. Les mesures ont donc été réalisées en faisant varier le nombre de processus MPI {process_text}, et en testant pour chaque cas les nombres de threads \\texttt{{numba}} suivants : {threads_text}. Les paramètres expérimentaux ont été repris de \\texttt{{results/experiment\\_environment.csv}}, soit le jeu de données \\texttt{{data/galaxy\\_5000}}, un pas de temps \\texttt{{0.0015}}, une grille \\texttt{{(15, 15, 1)}}, 30 itérations mesurées et 3 itérations d'échauffement.

Le temps mesuré correspond ici au coût d'une itération complète du calcul distribué, en incluant les synchronisations MPI nécessaires entre les processus. Le cas de référence est la configuration à un seul processus et un seul thread, utilisée pour calculer l'accélération de toutes les autres configurations.

{results_table}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{{time_figure_rel}}}
    \\caption{{Temps moyen par itération de \\texttt{{nbodies\\_grid\\_numba3.py}} en fonction du nombre de processus MPI et du nombre de threads Numba.}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{{speedup_figure_rel}}}
    \\caption{{Accélération de \\texttt{{nbodies\\_grid\\_numba3.py}} relativement au cas 1 processus / 1 thread.}}
\\end{{figure}}

La meilleure configuration mesurée est obtenue avec {int(best_row["processes"])} processus MPI et {int(best_row["threads"])} threads Numba, pour un temps moyen d'environ \\texttt{{{best_row["mean_iteration_ms"]:.2f} ms}} par itération et une accélération de \\texttt{{{best_speedup:.2f}x}}. Les résultats permettent ainsi d'observer l'effet combiné du parallélisme distribué et du parallélisme en mémoire partagée, ainsi que les limites introduites par les synchronisations MPI et par la concurrence entre threads lorsque le degré de parallélisme devient trop élevé sur la machine utilisée.

"""

    content = report_path.read_text(encoding="utf-8")
    section_marker = "\\section{MPI2 : Parallélisation du calcul}"
    start_marker = "\\subsection{Accélération obtenue}"
    start = content.index(start_marker, content.index(section_marker))
    end_marker = "%---------------------------------------------------"
    end = content.index(end_marker, start)
    updated = content[:start] + subsection_text + content[end:]
    report_path.write_text(updated, encoding="utf-8")


def driver(args: argparse.Namespace) -> None:
    environment = read_environment_csv(args.environment_csv.resolve())

    dataset = environment[("profilage", "jeu_de_donnees")]
    dt = float(environment[("profilage", "pas_de_temps")])
    cells = parse_cells(environment[("profilage", "grille")])
    frames = int(environment[("profilage", "iterations_mesurees")])
    warmup = int(environment[("profilage", "iterations_echauffement")])
    max_processes = args.max_processes or int(environment[("materiel", "coeurs_physiques")])
    max_threads = args.max_threads or int(environment[("materiel", "coeurs_physiques")])
    process_values = list(range(1, max_processes + 1))
    thread_values = sorted(set(args.threads_list)) if args.threads_list else list(range(1, max_threads + 1))

    rows: list[dict[str, float]] = []
    for processes in process_values:
        print(f"Mesures MPI2 en cours avec {processes} processus...", flush=True)
        process_rows = run_worker_subprocess(
            processes=processes,
            dataset=dataset,
            dt=dt,
            cells=cells,
            frames=frames,
            warmup=warmup,
            threads_list=thread_values,
        )
        rows.extend(process_rows)

    rows.sort(key=lambda row: (int(row["processes"]), int(row["threads"])))

    save_results_csv(args.output_csv.resolve(), rows, dataset, dt, cells)

    time_matrix, speedup_matrix = build_metric_matrices(rows, process_values, thread_values)
    create_heatmap(
        args.output_time_figure.resolve(),
        time_matrix,
        process_values,
        thread_values,
        "Temps moyen par iteration",
        "Temps (ms)",
        ".1f",
    )
    create_heatmap(
        args.output_speedup_figure.resolve(),
        speedup_matrix,
        process_values,
        thread_values,
        "Acceleration relative au cas 1 processus / 1 thread",
        "Acceleration",
        ".2f",
    )

    update_report(
        REPO_ROOT / "docs" / "examen" / "examen.tex",
        args.output_time_figure.resolve(),
        args.output_speedup_figure.resolve(),
        rows,
        process_values,
        thread_values,
    )

    baseline = next(
        row["mean_iteration_ms"]
        for row in rows
        if int(row["processes"]) == 1 and int(row["threads"]) == 1
    )
    best_row = min(rows, key=lambda row: row["mean_iteration_ms"])
    best_speedup = baseline / best_row["mean_iteration_ms"]

    print(f"CSV genere           : {args.output_csv.resolve()}")
    print(f"Figure temps generee : {args.output_time_figure.resolve()}")
    print(f"Figure accel generee : {args.output_speedup_figure.resolve()}")
    print(
        "Meilleur cas         : "
        f"{int(best_row['processes'])} processus, {int(best_row['threads'])} threads, "
        f"acceleration {best_speedup:.3f}x"
    )


def main() -> None:
    args = parse_args()

    if args.worker:
        if None in (args.dataset, args.dt, args.frames, args.warmup) or args.cells is None or not args.threads_list_worker:
            raise ValueError("Arguments incomplets pour le mode worker.")
        result = worker_measure(
            dataset=args.dataset,
            dt=args.dt,
            cells=tuple(args.cells),
            frames=args.frames,
            warmup=args.warmup,
            threads_list=args.threads_list_worker,
        )
        if result is not None:
            print(json.dumps(result))
        return

    driver(args)


if __name__ == "__main__":
    main()
