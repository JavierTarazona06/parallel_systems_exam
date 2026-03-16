#!/usr/bin/env python3
"""Prueba rápida para validar imports y uso básico de requirements.txt."""

from __future__ import annotations

import importlib
import sys
import traceback
from typing import Callable


def run_check(name: str, fn: Callable[[], None]) -> bool:
    print(f"[CHECK] {name}")
    try:
        fn()
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        return False
    print(f"[ OK ] {name}")
    return True


def check_numpy() -> None:
    import numpy as np

    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    dot = float(a @ b)
    if dot != 32.0:
        raise RuntimeError(f"Resultado inesperado de numpy: {dot}")


def check_numba() -> None:
    from numba import njit

    @njit
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 40)
    if result != 42:
        raise RuntimeError(f"Resultado inesperado de numba: {result}")


def check_llvmlite() -> None:
    import llvmlite
    import llvmlite.binding as llvm

    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    version = getattr(llvmlite, "__version__", None)
    if not version:
        raise RuntimeError("No se pudo leer la version de llvmlite")


def check_pyopengl() -> None:
    from OpenGL import GL  # noqa: F401

    # Valida que el acelerador C este disponible si fue instalado.
    importlib.import_module("OpenGL_accelerate")


def check_pysdl2() -> None:
    import sdl2

    rc = sdl2.SDL_Init(0)
    if rc != 0:
        msg = sdl2.SDL_GetError().decode("utf-8", errors="replace")
        raise RuntimeError(f"SDL_Init fallo: {msg}")
    sdl2.SDL_Quit()


def main() -> int:
    checks = [
        ("numpy", check_numpy),
        ("numba", check_numba),
        ("llvmlite", check_llvmlite),
        ("PyOpenGL / PyOpenGL_accelerate", check_pyopengl),
        ("PySDL2", check_pysdl2),
    ]

    passed = 0
    for name, fn in checks:
        if run_check(name, fn):
            passed += 1

    total = len(checks)
    print(f"\nResumen: {passed}/{total} checks pasaron.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
