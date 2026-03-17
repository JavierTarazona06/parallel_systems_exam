#!/usr/bin/env python3
"""
Mesure l'acceleration de nbodies_grid_numba1 en fonction du nombre de threads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ENVIRONMENT_CSV = REPO_ROOT / "results" / "experiment_environment.csv"
DEFAULT_RESULTS_CSV = REPO_ROOT / "results" / "nbodies1_acceleration.csv"
DEFAULT_RESULTS_PNG = REPO_ROOT / "docs" / "examen" / "imgs" / "nbodies1_acceleration.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mesure le temps de calcul et l'acceleration de nbodies_grid_numba1."
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
        "--output-figure",
        type=Path,
        default=DEFAULT_RESULTS_PNG,
        help="Figure PNG de sortie.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Nombre maximal de threads a tester. Par defaut, valeur lue depuis experiment_environment.csv.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Mode interne pour executer une seule mesure avec un nombre fixe de threads.",
    )
    parser.add_argument("--threads", type=int, help="Nombre de threads pour le mode worker.")
    parser.add_argument("--dataset", type=str, help="Jeu de donnees pour le mode worker.")
    parser.add_argument("--dt", type=float, help="Pas de temps pour le mode worker.")
    parser.add_argument("--frames", type=int, help="Nombre d'iterations mesurees pour le mode worker.")
    parser.add_argument("--warmup", type=int, help="Nombre d'iterations d'echauffement pour le mode worker.")
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


def worker_measure(
    threads: int,
    dataset: str,
    dt: float,
    cells: tuple[int, int, int],
    frames: int,
    warmup: int,
) -> dict[str, float]:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    import nbodies_grid_numba1

    system = nbodies_grid_numba1.NBodySystem(dataset, ncells_per_dir=cells)

    for _ in range(warmup):
        system.update_positions(dt)

    iteration_times = []
    for _ in range(frames):
        start = perf_counter()
        system.update_positions(dt)
        end = perf_counter()
        iteration_times.append((end - start) * 1000.0)

    measures = np.array(iteration_times, dtype=np.float64)
    result = {
        "threads": threads,
        "frames": int(frames),
        "warmup": int(warmup),
        "mean_iteration_ms": float(np.mean(measures)),
        "std_iteration_ms": float(np.std(measures)),
        "total_measured_ms": float(np.sum(measures)),
    }
    return result


def run_worker_subprocess(
    threads: int,
    dataset: str,
    dt: float,
    cells: tuple[int, int, int],
    frames: int,
    warmup: int,
) -> dict[str, float]:
    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = str(threads)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--threads",
        str(threads),
        "--dataset",
        dataset,
        "--dt",
        str(dt),
        "--frames",
        str(frames),
        "--warmup",
        str(warmup),
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
    )
    return json.loads(completed.stdout.strip())


def save_results_csv(output_csv: Path, rows: list[dict[str, float]], dataset: str, dt: float, cells: tuple[int, int, int]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        baseline = rows[0]["mean_iteration_ms"]
        for row in rows:
            speedup = baseline / row["mean_iteration_ms"]
            efficiency = speedup / row["threads"]
            writer.writerow(
                [
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


def create_figure(output_figure: Path, rows: list[dict[str, float]], dataset: str, dt: float, cells: tuple[int, int, int]) -> None:
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    threads = np.array([row["threads"] for row in rows], dtype=np.int32)
    mean_times = np.array([row["mean_iteration_ms"] for row in rows], dtype=np.float64)
    speedups = mean_times[0] / mean_times
    ideal = threads.astype(np.float64)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].plot(threads, mean_times, marker="o", linewidth=2, color="#c44e52")
    axes[0].set_title("Temps moyen de calcul")
    axes[0].set_xlabel("Nombre de threads")
    axes[0].set_ylabel("Temps par iteration (ms)")
    axes[0].set_xticks(threads)

    axes[1].plot(threads, speedups, marker="o", linewidth=2, color="#4c72b0", label="Mesure")
    axes[1].plot(threads, ideal, linestyle="--", color="#55a868", label="Ideal")
    axes[1].set_title("Acceleration obtenue")
    axes[1].set_xlabel("Nombre de threads")
    axes[1].set_ylabel("Acceleration")
    axes[1].set_xticks(threads)
    axes[1].legend()

    best_index = int(np.argmax(speedups))
    axes[1].text(
        0.98,
        0.05,
        f"Meilleure acceleration: {speedups[best_index]:.2f}x\navec {threads[best_index]} threads",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    fig.suptitle(
        (
            "Acceleration de nbodies_grid_numba1 en fonction du nombre de threads\n"
            f"donnees={Path(dataset).name}, grille={cells}, dt={dt}"
        ),
        fontsize=13,
    )
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)


def update_report(report_path: Path, figure_path: Path, rows: list[dict[str, float]]) -> None:
    baseline = rows[0]["mean_iteration_ms"]
    best_row = min(rows, key=lambda row: row["mean_iteration_ms"])
    best_speedup = baseline / best_row["mean_iteration_ms"]

    figure_rel = figure_path.relative_to(report_path.parent).as_posix()
    subsection_text = f"""\\subsection{{Accélération obtenue}}

L'accélération de la version parallélisée a été mesurée à l'aide du script \\texttt{{nbodies1\\_acceleration.py}} en faisant varier le nombre de threads \\texttt{{numba}} de 1 à 8, ce qui correspond au nombre de cœurs physiques disponibles sur la machine d'expérimentation. Les paramètres utilisés ont été repris du fichier \\texttt{{results/experiment\\_environment.csv}}, à savoir le jeu de données \\texttt{{data/galaxy\\_5000}}, un pas de temps \\texttt{{0.0015}}, une grille \\texttt{{(15, 15, 1)}}, 30 itérations mesurées et 3 itérations d'échauffement.

Les mesures portent uniquement sur le temps de calcul des trajectoires, sans prendre en compte l'affichage, afin d'évaluer directement l'effet de la parallélisation introduite par \\texttt{{parallel=True}} et \\texttt{{prange}}. L'accélération est calculée par rapport au temps moyen obtenu avec un seul thread.

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{{figure_rel}}}
    \\caption{{Temps moyen de calcul et accélération de \\texttt{{nbodies\\_grid\\_numba1.py}} en fonction du nombre de threads.}}
\\end{{figure}}

La figure montre que le temps moyen par itération diminue lorsque le nombre de threads augmente, ce qui confirme l'intérêt de la parallélisation \\texttt{{numba}} sur les boucles les plus coûteuses du calcul. La meilleure performance mesurée est obtenue avec {best_row["threads"]} threads, pour une accélération d'environ \\texttt{{{best_speedup:.2f}x}} par rapport à l'exécution mono-thread. L'accélération reste toutefois sous-linéaire, ce qui est cohérent avec la présence de parties demeurant séquentielles, avec les surcoûts de synchronisation entre threads et avec les accès mémoire partagés.

"""

    content = report_path.read_text(encoding="utf-8")
    start_marker = "\\subsection{Accélération obtenue}"
    end_marker = "%---------------------------------------------------"
    start = content.index(start_marker)
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
    max_threads = args.max_threads or int(environment[("materiel", "coeurs_physiques")])

    rows = []
    for threads in range(1, max_threads + 1):
        print(f"Mesure en cours avec {threads} thread(s)...", flush=True)
        row = run_worker_subprocess(
            threads=threads,
            dataset=dataset,
            dt=dt,
            cells=cells,
            frames=frames,
            warmup=warmup,
        )
        rows.append(row)

    save_results_csv(args.output_csv.resolve(), rows, dataset, dt, cells)
    create_figure(args.output_figure.resolve(), rows, dataset, dt, cells)
    update_report(REPO_ROOT / "docs" / "examen" / "examen.tex", args.output_figure.resolve(), rows)

    baseline = rows[0]["mean_iteration_ms"]
    best_row = min(rows, key=lambda row: row["mean_iteration_ms"])
    best_speedup = baseline / best_row["mean_iteration_ms"]

    print(f"CSV genere    : {args.output_csv.resolve()}")
    print(f"Figure generee: {args.output_figure.resolve()}")
    print(f"Meilleur cas  : {best_row['threads']} threads, acceleration {best_speedup:.3f}x")


def main() -> None:
    args = parse_args()

    if args.worker:
        if None in (args.threads, args.dataset, args.dt, args.frames, args.warmup) or args.cells is None:
            raise ValueError("Arguments incomplets pour le mode worker.")
        result = worker_measure(
            threads=args.threads,
            dataset=args.dataset,
            dt=args.dt,
            cells=tuple(args.cells),
            frames=args.frames,
            warmup=args.warmup,
        )
        print(json.dumps(result))
        return

    driver(args)


if __name__ == "__main__":
    main()
