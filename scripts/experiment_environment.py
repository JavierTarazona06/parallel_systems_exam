#!/usr/bin/env python3
"""
Collecte l'environnement d'execution des experiences et l'enregistre dans un CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enregistre l'environnement de la machine et les parametres du profilage."
    )
    parser.add_argument(
        "--dataset",
        default="data/galaxy_5000",
        help="Jeu de donnees utilise pour le profilage initial.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0015,
        help="Pas de temps utilise pour le profilage initial.",
    )
    parser.add_argument(
        "--cells",
        nargs=3,
        type=int,
        metavar=("NI", "NJ", "NK"),
        default=(15, 15, 1),
        help="Nombre de cellules de la grille pour le profilage initial.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Nombre d'iterations mesurees lors du profilage initial.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Nombre d'iterations d'echauffement du profilage initial.",
    )
    parser.add_argument(
        "--profile-image",
        default="docs/examen/imgs/profilage_initial.png",
        help="Chemin de l'image produite par le profilage initial.",
    )
    parser.add_argument(
        "--profile-csv",
        default="results/profilage_initial.csv",
        help="Chemin du CSV de mesures brutes du profilage initial.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "experiment_environment.csv",
        help="CSV de sortie pour l'environnement d'execution.",
    )
    return parser.parse_args()


def read_lscpu() -> dict[str, str]:
    try:
        output = subprocess.check_output(["lscpu"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return {}

    result: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def read_meminfo() -> tuple[str, str]:
    mem_total_kib = None
    swap_total_kib = None

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_total_kib = int(line.split()[1])
                elif line.startswith("SwapTotal:"):
                    swap_total_kib = int(line.split()[1])
    except OSError:
        return ("inconnu", "inconnu")

    return (format_kib(mem_total_kib), format_kib(swap_total_kib))


def format_kib(value: int | None) -> str:
    if value is None:
        return "inconnu"
    gib = value / (1024.0 * 1024.0)
    return f"{gib:.1f} GiB"


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "non_installe"


def collect_rows(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    lscpu = read_lscpu()
    memory_total, swap_total = read_meminfo()
    cells = tuple(args.cells)

    return [
        ("materiel", "systeme_exploitation", f"{platform.system()} {platform.release()}"),
        ("materiel", "plateforme", platform.platform()),
        ("materiel", "architecture", platform.machine()),
        ("materiel", "processeur", lscpu.get("Model name", platform.processor() or "inconnu")),
        ("materiel", "coeurs_physiques", lscpu.get("Core(s) per socket", "inconnu")),
        ("materiel", "coeurs_logiques", lscpu.get("CPU(s)", str(os.cpu_count() or "inconnu"))),
        ("materiel", "memoire_totale", memory_total),
        ("materiel", "swap_total", swap_total),
        ("logiciel", "interpreteur_python", sys.executable),
        ("logiciel", "version_python", platform.python_version()),
        ("logiciel", "version_numpy", package_version("numpy")),
        ("logiciel", "version_numba", package_version("numba")),
        ("logiciel", "version_matplotlib", package_version("matplotlib")),
        ("logiciel", "numba_num_threads", os.environ.get("NUMBA_NUM_THREADS", "non_defini")),
        ("profilage", "jeu_de_donnees", args.dataset),
        ("profilage", "pas_de_temps", str(args.dt)),
        ("profilage", "grille", str(cells)),
        ("profilage", "iterations_mesurees", str(args.frames)),
        ("profilage", "iterations_echauffement", str(args.warmup)),
        ("profilage", "image_generee", args.profile_image),
        ("profilage", "mesures_csv", args.profile_csv),
    ]


def write_csv(output_path: Path, rows: list[tuple[str, str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["categorie", "parametre", "valeur"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    write_csv(args.output.resolve(), rows)
    print(f"CSV genere : {args.output.resolve()}")


if __name__ == "__main__":
    main()
