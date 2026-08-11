#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificación del entorno de trabajo — Aprendizaje Automático y Grandes Datos (UNRaf)
====================================================================================

Se ejecuta en el LAB 1 y cada vez que algo deja de funcionar.

    python scripts/verificar_entorno.py            # verificación básica, sin red
    python scripts/verificar_entorno.py --datos    # además descarga todo el catálogo

Comprueba, en orden:
  1. Versión de Python.
  2. Presencia y versión de cada dependencia.
  3. Que el catálogo de datos sea importable y consistente.
  4. Que la aleatoriedad quede fijada de forma reproducible.
  5. Opcionalmente, que todos los conjuntos del catálogo se descarguen.

Salida: un informe legible y un código de salida distinto de cero si algo falla, para
que pueda usarse en integración continua.
"""
from __future__ import annotations

import argparse
import platform
import sys
from importlib import import_module

VERDE, ROJO, AMAR, FIN = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

PY_MIN = (3, 10)

PAQUETES = [
    ("numpy", "1.26"), ("pandas", "2.1"), ("scipy", "1.11"),
    ("sklearn", "1.4"), ("matplotlib", "3.8"), ("seaborn", "0.13"),
    ("plotly", "5.18"), ("xgboost", "2.0"), ("mlxtend", "0.23"),
    ("statsmodels", "0.14"), ("ucimlrepo", None), ("pyarrow", "14.0"),
]

fallas: list[str] = []
avisos: list[str] = []


def ok(msg):    print(f"  {VERDE}OK   {FIN} {msg}")
def error(msg): print(f"  {ROJO}FALLA{FIN} {msg}"); fallas.append(msg)
def aviso(msg): print(f"  {AMAR}AVISO{FIN} {msg}"); avisos.append(msg)


def _tupla(v: str):
    partes = []
    for x in v.split("."):
        num = "".join(c for c in x if c.isdigit())
        partes.append(int(num) if num else 0)
    return tuple(partes)


def paso_python():
    print("\n1. Intérprete de Python")
    v = sys.version_info
    if v[:2] >= PY_MIN:
        ok(f"Python {v.major}.{v.minor}.{v.micro} sobre {platform.system()} {platform.machine()}")
    else:
        error(f"Python {v.major}.{v.minor} es anterior al mínimo requerido "
              f"({PY_MIN[0]}.{PY_MIN[1]}). Actualizar el intérprete.")


def paso_paquetes():
    print("\n2. Dependencias")
    for nombre, minimo in PAQUETES:
        try:
            mod = import_module(nombre)
        except ImportError:
            error(f"{nombre}: no está instalado. Ejecutar: pip install -r requirements.txt")
            continue
        ver = getattr(mod, "__version__", "?")
        if minimo and ver != "?" and _tupla(ver) < _tupla(minimo):
            aviso(f"{nombre} {ver}: por debajo del mínimo sugerido ({minimo}).")
        else:
            ok(f"{nombre} {ver}")


def paso_catalogo():
    print("\n3. Catálogo de datos de la cátedra")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    try:
        from datos.catalogo import CATALOGO, listar, ruta_cache
    except Exception as e:
        error(f"No se pudo importar el catálogo: {type(e).__name__}: {e}")
        return
    ok(f"Catálogo importado: {len(CATALOGO)} conjuntos registrados")
    ok(f"Directorio de caché: {ruta_cache()}")

    sin_objetivo = [c.clave for c in CATALOGO.values()
                    if c.objetivo is None and c.tarea not in ("sin etiqueta", "transaccional")]
    if sin_objetivo:
        error(f"Conjuntos sin objetivo declarado y con tarea supervisada: {sin_objetivo}")
    else:
        ok("Todos los conjuntos supervisados declaran su variable objetivo")

    sin_licencia = [c.clave for c in CATALOGO.values() if not c.licencia]
    if sin_licencia:
        error(f"Conjuntos sin licencia declarada: {sin_licencia}")
    else:
        ok("Todos los conjuntos declaran licencia y citación")

    try:
        tabla = listar()
        assert len(tabla) == len(CATALOGO)
        ok(f"listar() devuelve {len(tabla)} filas")
    except Exception as e:
        error(f"listar() falló: {type(e).__name__}: {e}")


def paso_reproducibilidad():
    print("\n4. Reproducibilidad de la aleatoriedad")
    try:
        import numpy as np
        a = np.random.default_rng(42).normal(size=5)
        b = np.random.default_rng(42).normal(size=5)
        if np.array_equal(a, b):
            ok("Generador de NumPy con semilla fija: reproducible")
        else:
            error("El generador con la misma semilla dio resultados distintos.")

        from sklearn.model_selection import train_test_split
        X = np.arange(100).reshape(-1, 1)
        p1 = train_test_split(X, test_size=0.2, random_state=42)[0]
        p2 = train_test_split(X, test_size=0.2, random_state=42)[0]
        if np.array_equal(p1, p2):
            ok("Partición de scikit-learn con random_state fijo: reproducible")
        else:
            error("train_test_split con el mismo random_state dio particiones distintas.")
    except Exception as e:
        error(f"No se pudo verificar la reproducibilidad: {type(e).__name__}: {e}")


def paso_datos():
    print("\n5. Descarga del catálogo completo (requiere conexión)")
    try:
        from datos.catalogo import verificar_todos
    except Exception as e:
        error(f"No se pudo importar el catálogo: {e}")
        return
    tabla = verificar_todos(silencioso=True)
    for _, f in tabla.iterrows():
        if f["estado"] == "OK":
            ok(f"{f['clave']:12} {int(f['filas']):>9,} filas × {int(f['columnas']):>3} columnas  "
               f"huella {f['huella']}".replace(",", "."))
        else:
            error(f"{f['clave']:12} {f['detalle']}")


def main():
    ap = argparse.ArgumentParser(description="Verificación del entorno de la cátedra.")
    ap.add_argument("--datos", action="store_true",
                    help="además descarga y verifica todos los conjuntos del catálogo")
    args = ap.parse_args()

    print("=" * 78)
    print("VERIFICACIÓN DEL ENTORNO — Aprendizaje Automático y Grandes Datos — UNRaf")
    print("=" * 78)

    paso_python()
    paso_paquetes()
    paso_catalogo()
    paso_reproducibilidad()
    if args.datos:
        paso_datos()

    print("\n" + "=" * 78)
    if fallas:
        print(f"{ROJO}RESULTADO: {len(fallas)} falla(s).{FIN} El entorno no está listo.")
        for f in fallas:
            print(f"  · {f}")
    elif avisos:
        print(f"{AMAR}RESULTADO: entorno utilizable, con {len(avisos)} aviso(s).{FIN}")
    else:
        print(f"{VERDE}RESULTADO: entorno correcto.{FIN} "
              "Anotar esta salida en la bitácora del equipo.")
    print("=" * 78)
    sys.exit(1 if fallas else 0)


if __name__ == "__main__":
    main()
