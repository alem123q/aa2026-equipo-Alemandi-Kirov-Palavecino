#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""¿El sustituto sintético tiene las mismas columnas que el conjunto real?

POR QUÉ EXISTE ESTE ARCHIVO
---------------------------
El LAB 5 se caía con datos reales desde antes de que empezara el cuatrimestre. Su
contraejemplo de fuga usaba las columnas `casual` y `registered` de `bici`, que suman
exactamente la variable objetivo. Esas columnas NO existen en el conjunto que el
catálogo descarga: la versión de OpenML tiene 13 columnas y ninguna es ésa. Existían
en el SUSTITUTO SINTÉTICO, que se había escrito mirando otra versión del mismo
conjunto.

Y por eso nadie se enteró: `ejecutar_todos.py --sin-conexion` corría el cuaderno
entero sin un error, porque en modo sin conexión las columnas sí estaban. La prueba
de regresión pasaba justamente sobre los datos donde el error no se manifiesta.

Un sustituto que no comparte el esquema del original no sirve para probar cuadernos:
sirve para probar el sustituto. Este control cierra ese agujero.

QUÉ COMPRUEBA
-------------
1. Que el sustituto tenga exactamente las mismas columnas que el conjunto real, en el
   mismo orden y con la misma familia de tipo (numérico, categórico, booleano, fecha).
2. Que ningún nombre de columna que aparezca escrito en los módulos de laboratorio
   falte en el conjunto real o en el sustituto.

QUÉ NO PUEDE COMPROBAR
----------------------
Un conjunto que no esté descargado no se puede comparar contra nada, y se informa
como no verificable en lugar de darse por bueno. Con `--estricto` eso también corta,
que es lo que corresponde antes de dictar.

    python datos/verificar_sustitutos.py
    python datos/verificar_sustitutos.py --estricto
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
CACHE = AQUI / "cache"
LABS = AQUI.parent / "scripts" / "labs"


def familia(s: pd.Series) -> str:
    t = str(s.dtype)
    if t == "bool":
        return "booleano"
    if t.startswith("datetime"):
        return "fecha"
    if s.dtype.kind in "if":
        return "numérico"
    return "categórico"


def esquema(ruta: Path):
    d = pd.read_parquet(ruta)
    return list(d.columns), {c: familia(d[c]) for c in d.columns}


def comparar():
    """Devuelve (fallas, no_verificables, comparados)."""
    fallas, no_verificables, comparados = [], [], 0
    for real in sorted(CACHE.glob("*.parquet")):
        if real.name.endswith(".offline.parquet"):
            continue
        clave = real.stem
        sust = CACHE / f"{clave}.offline.parquet"
        if not sust.exists():
            no_verificables.append(f"{clave}: no hay sustituto generado")
            continue
        cols_r, tip_r = esquema(real)
        cols_s, tip_s = esquema(sust)
        comparados += 1

        solo_real = [c for c in cols_r if c not in cols_s]
        solo_sust = [c for c in cols_s if c not in cols_r]
        if solo_real or solo_sust:
            fallas.append(
                f"«{clave}»: el sustituto no tiene el mismo esquema.\n"
                f"    sólo en el real      : {solo_real or '—'}\n"
                f"    sólo en el sustituto : {solo_sust or '—'}\n"
                f"    Un cuaderno que nombre una columna de una lista y no de la otra\n"
                f"    corre en un modo y se cae en el otro.")
            continue
        if cols_r != cols_s:
            fallas.append(f"«{clave}»: mismas columnas, distinto orden.\n"
                          f"    real      : {cols_r}\n"
                          f"    sustituto : {cols_s}")
            continue
        distintos = [f"{c} ({tip_r[c]} contra {tip_s[c]})"
                     for c in cols_r if tip_r[c] != tip_s[c]]
        if distintos:
            fallas.append(f"«{clave}»: la familia de tipo no coincide en {len(distintos)} "
                          f"columna(s).\n    " + "\n    ".join(distintos))
    # Conjuntos del catálogo que no están descargados: no se pueden comparar.
    for sust in sorted(CACHE.glob("*.offline.parquet")):
        clave = sust.name[:-len(".offline.parquet")]
        if not (CACHE / f"{clave}.parquet").exists():
            no_verificables.append(f"{clave}: falta el conjunto real, no hay contra qué comparar")
    return fallas, no_verificables, comparados


# Nombres de columna escritos a mano en el código de los laboratorios. Se buscan en los
# contextos donde un literal ES un nombre de columna, y no en cualquier cadena suelta.
PATRONES = [
    # Incluye las comillas escapadas de una f-string:  f'{ret[\"Customer ID\"]}'
    re.compile(r"\[\\?['\"]([A-Za-z_][\w\s().\-]*)\\?['\"]\]"),      #  df['col']
    re.compile(r"columns=\[([^\]]+)\]"),                            #  columns=['a','b']
    re.compile(r"\.pop\(['\"]([\w\s().\-]+)['\"]\)"),               #  df.pop('col')
    re.compile(r"sort_values\(['\"]([\w\s().\-]+)['\"]\)"),
]
CARGA = re.compile(r"cargar\(\s*['\"](\w+)['\"]")


def columnas_de_los_labs():
    """Para cada módulo de laboratorio: qué conjuntos carga y qué columnas nombra."""
    out = {}
    for mod in sorted(LABS.glob("lab*.py")):
        txt = mod.read_text(encoding="utf8")
        claves = sorted(set(CARGA.findall(txt)))
        nombres = set()
        for p in PATRONES[:1] + PATRONES[2:]:
            nombres |= set(p.findall(txt))
        for grupo in PATRONES[1].findall(txt):
            nombres |= set(re.findall(r"['\"]([\w\s().\-]+)['\"]", grupo))
        out[mod.stem] = (claves, sorted(n for n in nombres if n))
    return out


def revisar_labs():
    # `avisos` son hallazgos concluyentes; `dudosos`, los que no se pueden decidir
    # porque el laboratorio también usa un conjunto que no está descargado.
    avisos, dudosos = [], []
    for lab, (claves, nombres) in columnas_de_los_labs().items():
        if not claves or not nombres:
            continue
        conocidas_r, conocidas_s, faltan_datos = set(), set(), []
        for k in claves:
            r, s = CACHE / f"{k}.parquet", CACHE / f"{k}.offline.parquet"
            if r.exists():
                conocidas_r |= set(pd.read_parquet(r).columns)
            else:
                faltan_datos.append(k)
            if s.exists():
                conocidas_s |= set(pd.read_parquet(s).columns)
        # Si el laboratorio carga además algún conjunto que no está descargado, el aviso
        # sigue valiendo pero no es concluyente: la columna podría estar en ése. Se
        # informa con la salvedad en lugar de callarlo, que es lo que hacía antes y por
        # eso tres laboratorios rotos pasaron sin que nada los señalara.
        salvedad = (f"  (sin verificar contra {', '.join(faltan_datos)}, que no está descargado)"
                    if faltan_datos else "")
        for n in nombres:
            en_r, en_s = n in conocidas_r, n in conocidas_s
            if en_r and not en_s:
                destino = dudosos if faltan_datos else avisos
                destino.append(f"{lab}: «{n}» existe en el conjunto real y no en el "
                               f"sustituto: el cuaderno se cae en modo sin conexión.{salvedad}")
            elif en_s and not en_r:
                destino = dudosos if faltan_datos else avisos
                destino.append(f"{lab}: «{n}» existe SÓLO en el sustituto sintético: el "
                               f"cuaderno se cae con datos reales, que es como se dicta.{salvedad}")
    return avisos, dudosos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estricto", action="store_true",
                    help="también corta si algún conjunto no se pudo verificar")
    args = ap.parse_args()

    fallas, no_verificables, comparados = comparar()
    avisos, dudosos = revisar_labs()

    print(f"esquemas comparados: {comparados}")
    if fallas:
        print(f"\nESQUEMA — {len(fallas)} conjunto(s) donde el sustituto no reproduce el real\n")
        for f in fallas:
            print("  " + f.replace("\n", "\n  ") + "\n")
    else:
        print("  todos los sustituidos reproducen el esquema real")

    if avisos:
        print(f"\nCUADERNOS — {len(avisos)} columna(s) que existen en un modo y no en el otro\n")
        for a in avisos:
            print("  " + a)
    else:
        print("  ningún laboratorio nombra una columna que falte en alguno de los dos modos")

    if dudosos:
        print(f"\nSIN DECIDIR — {len(dudosos)} nombre(s) de un laboratorio que también usa "
              f"un conjunto no descargado.\n  Puede ser una columna correcta de ése. Se listan "
              f"para que no queden invisibles.\n")
        for d in dudosos:
            print("  " + d)

    if no_verificables:
        print(f"\nNO VERIFICABLES ({len(no_verificables)}):")
        for x in no_verificables:
            print("  " + x)

    malo = bool(fallas or avisos) or (args.estricto and bool(no_verificables))
    return 1 if malo else 0


if __name__ == "__main__":
    sys.exit(main())
