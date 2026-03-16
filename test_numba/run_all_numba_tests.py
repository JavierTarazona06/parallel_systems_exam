#!/usr/bin/env python3
"""Runner de pruebas para scripts de test_numba.

Ejecuta los scripts solicitados y reporta errores con detalle.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    script: str
    mode: str  # "must_exit" o "must_start"
    startup_timeout: float = 0.0


@dataclass
class Result:
    target: Target
    ok: bool
    duration_s: float
    returncode: int | None
    note: str
    stdout: str
    stderr: str


TARGETS: list[Target] = [
    Target("mandelbrot.py", "must_exit"),
    Target("mandelbrot_numba.py", "must_exit"),
    Target("visualizer3d_vbo.py", "must_start", startup_timeout=5.0),
    Target("visualizer3d_vbo.py", "must_start", startup_timeout=5.0),
]


def tail_lines(text: str, max_lines: int = 25) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def run_must_exit(target: Target, python_bin: str, script_dir: Path, env: dict[str, str]) -> Result:
    cmd = [python_bin, str(script_dir / target.script)]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=script_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    duration = time.perf_counter() - start
    return Result(
        target=target,
        ok=proc.returncode == 0,
        duration_s=duration,
        returncode=proc.returncode,
        note="termino correctamente" if proc.returncode == 0 else "fallo al terminar",
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def terminate_process(proc: subprocess.Popen[str], timeout_s: float = 3.0) -> tuple[str, str]:
    try:
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return stdout, stderr
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            return stdout, stderr
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return stdout, stderr


def run_must_start(target: Target, python_bin: str, script_dir: Path, env: dict[str, str]) -> Result:
    cmd = [python_bin, str(script_dir / target.script)]
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=script_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.perf_counter() + target.startup_timeout
    while time.perf_counter() < deadline:
        code = proc.poll()
        if code is not None:
            stdout, stderr = proc.communicate()
            duration = time.perf_counter() - start
            return Result(
                target=target,
                ok=code == 0,
                duration_s=duration,
                returncode=code,
                note="salio antes del timeout de inicio",
                stdout=stdout,
                stderr=stderr,
            )
        time.sleep(0.1)

    stdout, stderr = terminate_process(proc)
    duration = time.perf_counter() - start
    return Result(
        target=target,
        ok=True,
        duration_s=duration,
        returncode=0,
        note=f"inicio bien y siguio corriendo {target.startup_timeout:.1f}s",
        stdout=stdout,
        stderr=stderr,
    )


def print_failure_details(result: Result) -> None:
    if result.stdout.strip():
        print("  stdout (ultimas lineas):")
        print(tail_lines(result.stdout))
    if result.stderr.strip():
        print("  stderr (ultimas lineas):")
        print(tail_lines(result.stderr))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ejecuta pruebas para scripts en test_numba.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Ejecutable de Python a usar (por defecto: el actual).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=5.0,
        help="Timeout (s) para scripts en modo must_start.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    # Evita bloqueo de plt.show() en entornos sin GUI para los scripts mandelbrot.
    env.setdefault("MPLBACKEND", "Agg")

    results: list[Result] = []
    for i, base_target in enumerate(TARGETS, start=1):
        target = base_target
        if target.mode == "must_start":
            target = Target(target.script, target.mode, args.startup_timeout)

        label = f"{target.script} (run {i})"
        print(f"[RUN ] {label} | mode={target.mode}")

        if not (script_dir / target.script).exists():
            result = Result(
                target=target,
                ok=False,
                duration_s=0.0,
                returncode=None,
                note="archivo no encontrado",
                stdout="",
                stderr="",
            )
        elif target.mode == "must_exit":
            result = run_must_exit(target, args.python, script_dir, env)
        elif target.mode == "must_start":
            result = run_must_start(target, args.python, script_dir, env)
        else:
            result = Result(
                target=target,
                ok=False,
                duration_s=0.0,
                returncode=None,
                note=f"modo invalido: {target.mode}",
                stdout="",
                stderr="",
            )

        results.append(result)
        status = "OK" if result.ok else "FAIL"
        rc = f" rc={result.returncode}" if result.returncode is not None else ""
        print(f"[{status:4}] {label}{rc} | {result.duration_s:.2f}s | {result.note}")

        if not result.ok:
            print_failure_details(result)
        print()

    failed = [r for r in results if not r.ok]
    passed = len(results) - len(failed)

    print("==== RESUMEN ====")
    print(f"Total:  {len(results)}")
    print(f"OK:     {passed}")
    print(f"FAIL:   {len(failed)}")

    if failed:
        print("Fallaron:")
        for r in failed:
            print(f"- {r.target.script} | {r.note} | rc={r.returncode}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
