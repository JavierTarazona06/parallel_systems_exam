#!/usr/bin/env python3
"""
Profile le temps passé dans le calcul et dans l'affichage pour nbodies_grid_numba.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import nbodies_grid_numba
import visualizer3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile le temps de calcul et d'affichage de nbodies_grid_numba."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "galaxy_5000",
        help="Jeu de donnees a utiliser.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0015,
        help="Pas de temps utilise pour la simulation.",
    )
    parser.add_argument(
        "--cells",
        nargs=3,
        type=int,
        metavar=("NI", "NJ", "NK"),
        default=(15, 15, 1),
        help="Nombre de cellules de la grille.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=40,
        help="Nombre d'iterations profilees apres echauffement.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Nombre d'iterations d'echauffement pour amortir la compilation JIT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "examen" / "imgs" / "profilage_initial.png",
        help="Fichier PNG de sortie pour la figure.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "profilage_initial.csv",
        help="Fichier CSV de sortie pour les mesures brutes.",
    )
    return parser.parse_args()


def build_visualizer(system: nbodies_grid_numba.NBodySystem) -> visualizer3d.Visualizer3D:
    bounds = [
        [system.box[0][0], system.box[1][0]],
        [system.box[0][1], system.box[1][1]],
        [system.box[0][2], system.box[1][2]],
    ]
    intensities = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    return visualizer3d.Visualizer3D(system.positions, system.colors, intensities, bounds)


def run_profile(
    dataset: Path,
    dt: float,
    cells: tuple[int, int, int],
    frames: int,
    warmup: int,
) -> dict[str, np.ndarray]:
    system = nbodies_grid_numba.NBodySystem(str(dataset), ncells_per_dir=cells)
    visualizer = build_visualizer(system)

    display_ms = []
    compute_ms = []
    sync_ms = []

    try:
        for _ in range(warmup):
            if not visualizer._handle_events():
                raise RuntimeError("La fenetre a ete fermee pendant l'echauffement.")
            visualizer._render()
            system.update_positions(dt)
            visualizer.update_points(system.positions)

        for _ in range(frames):
            t0 = perf_counter()
            running = visualizer._handle_events()
            if not running:
                break
            visualizer._render()
            t1 = perf_counter()

            system.update_positions(dt)
            t2 = perf_counter()

            visualizer.update_points(system.positions)
            t3 = perf_counter()

            display_ms.append((t1 - t0) * 1000.0)
            compute_ms.append((t2 - t1) * 1000.0)
            sync_ms.append((t3 - t2) * 1000.0)
    finally:
        visualizer.cleanup()

    if not display_ms:
        raise RuntimeError("Aucune iteration mesuree. La fenetre a probablement ete fermee trop tot.")

    return {
        "display_ms": np.array(display_ms, dtype=np.float64),
        "compute_ms": np.array(compute_ms, dtype=np.float64),
        "sync_ms": np.array(sync_ms, dtype=np.float64),
    }


def save_csv(csv_path: Path, display_ms: np.ndarray, compute_ms: np.ndarray, sync_ms: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "display_render_ms",
                "display_sync_ms",
                "display_total_ms",
                "compute_ms",
            ]
        )
        for index, (display_value, sync_value, compute_value) in enumerate(
            zip(display_ms, sync_ms, compute_ms),
            start=1,
        ):
            writer.writerow(
                [
                    index,
                    f"{display_value:.6f}",
                    f"{sync_value:.6f}",
                    f"{display_value + sync_value:.6f}",
                    f"{compute_value:.6f}",
                ]
            )


def create_figure(
    output_path: Path,
    dataset: Path,
    cells: tuple[int, int, int],
    dt: float,
    display_ms: np.ndarray,
    compute_ms: np.ndarray,
    sync_ms: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_total_ms = display_ms + sync_ms
    iterations = np.arange(1, len(display_ms) + 1)
    mean_display = float(np.mean(display_total_ms))
    mean_compute = float(np.mean(compute_ms))
    total = mean_display + mean_compute

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].plot(iterations, compute_ms, label="Calcul", color="#c44e52", linewidth=2)
    axes[0].plot(
        iterations,
        display_total_ms,
        label="Affichage",
        color="#4c72b0",
        linewidth=2,
    )
    axes[0].set_title("Temps par iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Temps (ms)")
    axes[0].legend()

    categories = ["Affichage", "Calcul"]
    means = [mean_display, mean_compute]
    colors = ["#4c72b0", "#c44e52"]
    bars = axes[1].bar(categories, means, color=colors, width=0.6)
    axes[1].set_title("Temps moyen par iteration")
    axes[1].set_ylabel("Temps (ms)")

    for bar, value in zip(bars, means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.1f} ms",
            ha="center",
            va="bottom",
        )

    axes[1].text(
        0.5,
        0.92,
        (
            f"Affichage: {100.0 * mean_display / total:.1f}%\n"
            f"Calcul: {100.0 * mean_compute / total:.1f}%"
        ),
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    fig.suptitle(
        (
            "Profilage initial de nbodies_grid_numba\n"
            f"donnees={dataset.name}, grille={cells}, dt={dt}, iterations={len(display_ms)}"
        ),
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    cells = tuple(args.cells)

    measures = run_profile(
        dataset=dataset,
        dt=args.dt,
        cells=cells,
        frames=args.frames,
        warmup=args.warmup,
    )

    save_csv(
        csv_path=args.csv.resolve(),
        display_ms=measures["display_ms"],
        compute_ms=measures["compute_ms"],
        sync_ms=measures["sync_ms"],
    )
    create_figure(
        output_path=args.output.resolve(),
        dataset=dataset,
        cells=cells,
        dt=args.dt,
        display_ms=measures["display_ms"],
        compute_ms=measures["compute_ms"],
        sync_ms=measures["sync_ms"],
    )

    display_total_ms = measures["display_ms"] + measures["sync_ms"]
    compute_ms = measures["compute_ms"]
    mean_display = float(np.mean(display_total_ms))
    mean_compute = float(np.mean(compute_ms))
    dominant = "calcul" if mean_compute > mean_display else "affichage"

    print(f"Profil termine sur {len(compute_ms)} iterations.")
    print(f"Temps moyen affichage : {mean_display:.3f} ms")
    print(f"Temps moyen calcul    : {mean_compute:.3f} ms")
    print(f"Partie dominante      : {dominant}")
    print(f"Figure enregistree    : {args.output.resolve()}")
    print(f"Mesures CSV           : {args.csv.resolve()}")


if __name__ == "__main__":
    main()
