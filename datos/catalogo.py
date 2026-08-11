"""
Catálogo de conjuntos de datos de la cátedra
============================================
Aprendizaje Automático y Grandes Datos — UNRaf

Este módulo es la ÚNICA vía por la que los cuadernos de la cátedra acceden a datos.
Ningún cuaderno debe contener una URL, una ruta absoluta ni una lectura directa de archivo.

Reglas de diseño
----------------
1. Fuentes inmutables. Se prioriza OpenML por `data_id` (versionado e inmutable) y el
   repositorio UCI por `id`. Se descartan fuentes que puedan cambiar sin aviso.
2. Sin cuentas personales. Ningún dataset depende de un Drive, una hoja de cálculo
   publicada ni una credencial.
3. Caché local. La primera lectura descarga; las siguientes leen del disco. El curso
   funciona sin conexión a partir de la segunda ejecución.
4. Huella verificable. Cada carga comprueba forma, columnas y tipos contra el manifiesto
   de la cátedra. Si la fuente cambió, el error aparece al cargar y no tres celdas después.
5. Agnóstico de infraestructura. Funciona igual en instalación local, en contenedor y en
   servicios de cuadernos en la nube. El directorio de caché se resuelve solo.

Uso
---
    from datos.catalogo import cargar, listar, ficha

    listar()                      # tabla con todos los conjuntos disponibles
    ficha("banco")                # descripción, licencia, variable objetivo y citación
    df = cargar("banco")          # DataFrame listo para trabajar
    X, y = cargar("banco", xy=True)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = ["cargar", "listar", "ficha", "ruta_cache", "verificar_todos", "CATALOGO"]

VERSION_CATALOGO = "2026.1"


# --------------------------------------------------------------------------- registro
@dataclass(frozen=True)
class Conjunto:
    clave: str
    nombre: str
    fuente: str                      # "openml" | "uci" | "url"
    ref: str | int                   # data_id de OpenML, id de UCI o URL
    objetivo: str | None
    tarea: str                       # "regresión" | "clasificación" | "sin etiqueta" | "transaccional"
    filas: int | None
    columnas: int | None
    licencia: str
    citacion: str
    usos: tuple[str, ...]
    nota: str
    lectura: dict = field(default_factory=dict)   # parámetros extra de lectura
    advertencia: str = ""


CATALOGO: dict[str, Conjunto] = {

    "bici": Conjunto(
        clave="bici",
        nombre="Bike Sharing Demand — demanda horaria de bicicletas compartidas",
        fuente="openml", ref=42712, objetivo="count", tarea="regresión",
        filas=17379, columnas=13,
        licencia="CC BY 4.0 (fuente original: UCI, id 275)",
        citacion=("Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble "
                  "detectors and background knowledge. Progress in Artificial Intelligence, "
                  "2(2-3), 113-127. https://doi.org/10.1007/s13748-013-0040-3"),
        usos=("LAB 7", "LAB 9", "LAB 6", "TP2 (demostración)"),
        nota=("Mezcla real de variables numéricas y categóricas, con estructura temporal de dos "
              "años. Es el conjunto de referencia para regresión en la asignatura."),
        # ADVERTENCIA CORREGIDA. La anterior decía que este conjunto trae las columnas
        # `casual` y `registered`, que suman exactamente `count`, y ofrecía un
        # parámetro `incluir_fuga=True` para conservarlas. Es falso para los datos que
        # el catálogo declara y descarga: la versión de OpenML 42712 tiene 13 columnas
        # y ninguna de las dos está. Las tenía el sustituto sintético que se genera sin
        # conexión, y de ahí venía la confusión. El contraejemplo de fuga que dependía
        # de ellas se rehizo sobre `banco`, donde la fuga es real y está en los datos.
        advertencia=("ATENCIÓN DIDÁCTICA: las filas son horas consecutivas de dos años y vienen "
                     "en orden cronológico. Una partición aleatoria pone el futuro en el "
                     "entrenamiento y el pasado en la prueba, que es fuga temporal: el modelo "
                     "rellena huecos entre horas conocidas en lugar de predecir horas nuevas. "
                     "Para estimar desempeño, partir por tiempo y no al azar."),
    ),

    "banco": Conjunto(
        clave="banco",
        nombre="Bank Marketing — campañas telefónicas de un banco portugués",
        fuente="openml", ref=1461, objetivo="Class", tarea="clasificación",
        filas=45211, columnas=17,
        licencia="CC BY 4.0 (fuente original: UCI, id 222)",
        citacion=("Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict "
                  "the success of bank telemarketing. Decision Support Systems, 62, 22-31. "
                  "https://doi.org/10.1016/j.dss.2014.03.001"),
        usos=("LAB 8", "LAB 9", "LAB 10"),
        nota=("11,7 % de clase positiva. Un clasificador que responde siempre «no» alcanza "
              "88,3 % de exactitud: es el caso con el que se demuestra por qué la exactitud "
              "no alcanza y por qué toda comparación necesita una línea de base."),
        advertencia=("Se accede por OpenML y no por UCI a propósito: el ZIP de UCI contiene "
                     "cuatro archivos distintos (bank.csv, bank-full.csv, bank-additional.csv y "
                     "bank-additional-full.csv) con distinta cantidad de filas y columnas, lo que "
                     "produce resultados no comparables entre equipos."),
    ),

    "autos": Conjunto(
        clave="autos",
        nombre="Don't Get Kicked — compra de autos usados en subasta",
        fuente="openml", ref=41162, objetivo="IsBadBuy", tarea="clasificación",
        filas=72983, columnas=33,
        licencia="CC0 (dominio público, declarado en OpenML)",
        citacion=("Carvana. (2011). Don't Get Kicked! [Conjunto de datos]. Kaggle. "
                  "Publicado en OpenML con data_id 41162."),
        usos=("LAB 4", "LAB 11"),
        nota=("El 95,5 % de las filas tiene algún valor faltante, de un proceso comercial real y "
              "no inyectados artificialmente. Mixto, desbalanceado (12,3 % positiva) y grande: "
              "es donde los métodos de ensamble muestran su ventaja práctica."),
    ),

    "aire": Conjunto(
        clave="aire",
        nombre="Air Quality — mediciones horarias de un sensor de calidad de aire",
        fuente="uci", ref=360, objetivo="C6H6(GT)", tarea="regresión",
        filas=9358, columnas=15,
        licencia="CC BY 4.0",
        citacion=("De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008). "
                  "On field calibration of an electronic nose for benzene estimation. Sensors and "
                  "Actuators B: Chemical, 129(2), 750-757. "
                  "https://doi.org/10.1016/j.snb.2007.09.060"),
        usos=("LAB 4",),
        nota=("El mejor ejemplo disponible de faltante disfrazado: los valores ausentes vienen "
              "codificados como -200, no como nulos. Quien no lea la documentación calcula medias "
              "sin sentido y no se entera. Además usa coma decimal."),
        advertencia=("La función de carga NO convierte los -200 automáticamente. Es parte del "
                     "ejercicio que el estudiante los descubra."),
    ),

    "porotos": Conjunto(
        clave="porotos",
        nombre="Dry Bean — clasificación de siete variedades de poroto por morfometría",
        fuente="uci", ref=602, objetivo="Class", tarea="clasificación",
        filas=13611, columnas=17,
        licencia="CC BY 4.0",
        citacion=("Koklu, M., & Ozkan, I. A. (2020). Multiclass classification of dry beans using "
                  "computer vision and machine learning techniques. Computers and Electronics in "
                  "Agriculture, 174, 105507. https://doi.org/10.1016/j.compag.2020.105507"),
        usos=("LAB 10",),
        nota=("Siete clases moderadamente desbalanceadas y características geométricas muy "
              "correlacionadas entre sí. Sirve para matriz de confusión multiclase, promedios "
              "macro y micro, y para mostrar por qué el análisis de componentes principales tiene "
              "sentido cuando las variables son redundantes."),
    ),

    # CAMBIO DE VÍA DE ACCESO, comprobado. Este conjunto se declaraba con
    # `fuente="uci"` e `id=502`, y por ahí NO SE PUEDE BAJAR: `ucimlrepo` sólo sirve
    # los conjuntos que UCI marcó como importables, y éste es un comprimido con
    # varios archivos adentro. El error era «exists in the repository, but is not
    # available for import», y le iba a pasar a cualquier alumno.
    #
    # El identificador de OpenML se buscó en una máquina con acceso y se comprobó
    # contra la forma que este mismo catálogo declara: 1.067.371 filas × 8 columnas.
    "retail": Conjunto(
        clave="retail",
        nombre="Online Retail II — transacciones de un comercio electrónico británico",
        fuente="openml", ref=43368, objetivo=None, tarea="transaccional",
        filas=1067371, columnas=8,
        licencia="CC BY 4.0",
        citacion=("Chen, D. (2019). Online Retail II [Conjunto de datos]. UCI Machine Learning "
                  "Repository. https://doi.org/10.24432/C5CG6D"),
        usos=("LAB 2", "Unidad 8 — reglas de asociación"),
        nota=("Dos períodos (2009-2010 y 2010-2011) con el mismo esquema, lo que permite enseñar "
              "integración de fuentes homogéneas. Agrupando por número de factura se obtiene la "
              "lista de ítems por transacción para A priori y Eclat."),
        advertencia=("Requiere limpieza antes de aplicar reglas de asociación: hay cancelaciones "
                     "con cantidad negativa, y un mismo código de producto aparece con "
                     "descripciones distintas. Esa limpieza es parte del ejercicio."),
    ),

    # De dónde salió. El cuaderno de ejercicios de pandas de la cátedra trae 41
    # consignas resueltas sobre este archivo y el conjunto no estaba en el catálogo:
    # el cuaderno leía una ruta de Colab. Es un conjunto argentino, público, con
    # licencia y con un portal que lo declara «histórico», o sea sin actualizaciones
    # futuras previstas, que es exactamente lo que el catálogo prefiere.
    #
    # LO QUE NO SE PUDO COMPROBAR: que el archivo del portal sea byte a byte el que
    # está en la carpeta de la cátedra. La huella se registró sobre el archivo de la
    # cátedra; si la descarga difiere, el control de integridad avisa al cargar, que
    # es para lo que existe.
    "traslados": Conjunto(
        clave="traslados",
        nombre="Traslados COVID-19 — viajes sanitarios en la Ciudad de Buenos Aires",
        fuente="url",
        ref=("https://data.buenosaires.gob.ar/dataset/traslados-covid-19/resource/"
             "48970b40-5d73-4995-a69a-54124a1ed22c/download"),
        objetivo=None, tarea="transaccional",
        filas=76362, columnas=7,
        licencia="CC BY 2.5 AR",
        citacion=("Secretaría de Transporte y Obras Públicas, Jefatura de Gabinete de Ministros, "
                  "Gobierno de la Ciudad de Buenos Aires. (2021). Traslados COVID-19 "
                  "[Conjunto de datos]. data.buenosaires.gob.ar"),
        usos=("LAB 2", "Unidad 2 — manipulación de datos"),
        nota=("76.362 traslados en taxi y ambulancia a unidades febriles, barrios y hoteles "
              "entre abril de 2020 y julio de 2021. Es el conjunto de ejercitación de pandas: "
              "una fila por traslado, dos columnas categóricas chicas y tres con ausencias "
              "masivas, que es la forma en la que llegan los registros administrativos."),
        lectura={"encoding": "utf-8-sig"},
        advertencia=("Dos trampas reales. La fecha viene en formato SAS —«07APR2020:00:00:00», "
                     "con el mes abreviado en inglés— y exige "
                     "format='%d%b%Y:%H:%M:%S'; sin eso la conversión no es confiable. Y tres "
                     "columnas tienen entre 90 y 97 por ciento de ausencias, así que cualquier "
                     "agrupación por oficina, CESAC o recorrido describe una minoría del "
                     "conjunto y hay que decirlo."),
    ),

    # PENDIENTE, MISMO PROBLEMA QUE `retail` Y TODAVÍA SIN RESOLVER: el id 240 de UCI
    # tampoco es importable por `ucimlrepo`, y una búsqueda en OpenML por «human
    # activity» y por «har» no devolvió ningún candidato con esta forma. Hasta que
    # aparezca un identificador COMPROBADO, este conjunto no se puede cargar y ningún
    # material puede depender de él.
    "actividad": Conjunto(
        clave="actividad",
        nombre="Human Activity Recognition — señales de acelerómetro de teléfonos",
        fuente="uci", ref=240, objetivo="Activity", tarea="sin etiqueta",
        filas=10299, columnas=562,
        licencia="CC BY 4.0",
        citacion=("Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). "
                  "A public domain dataset for human activity recognition using smartphones. "
                  "ESANN 2013 Proceedings, 437-442."),
        usos=("Unidad 8 — agrupamiento",),
        nota=("561 variables con estructura de variedad en alta dimensión, nada esférica. Las "
              "actividades estáticas se solapan y las dinámicas se separan, de modo que K-medias "
              "falla de manera instructiva frente al agrupamiento jerárquico o por densidad. Las "
              "etiquetas se ocultan durante el ejercicio y se usan al final para calcular el "
              "índice de Rand ajustado."),
    ),

    "adultos": Conjunto(
        clave="adultos",
        nombre="Adult — ingresos a partir del censo de los Estados Unidos de 1994",
        fuente="openml", ref=1590, objetivo="class", tarea="clasificación",
        filas=48842, columnas=15,
        licencia="CC BY 4.0 (fuente original: UCI, id 2)",
        citacion=("Becker, B., & Kohavi, R. (1996). Adult [Conjunto de datos]. UCI Machine "
                  "Learning Repository. https://doi.org/10.24432/C5XW20"),
        usos=("LAB 5", "Unidad 1 — ética"),
        nota=("Mezcla de numéricas y categóricas con faltantes codificados como «?», es decir "
              "disfrazados de categoría válida. Es el conjunto de referencia para construir "
              "flujos con transformación por columnas."),
        advertencia=("Contiene atributos sensibles (raza, sexo, país de origen). Se usa "
                     "deliberadamente en la unidad de ética para analizar equidad del "
                     "clasificador, no solo su exactitud."),
    ),

    "diamantes": Conjunto(
        clave="diamantes",
        nombre="Diamonds — precio y características de 54.000 diamantes",
        fuente="openml", ref=42225, objetivo="price", tarea="regresión",
        filas=53940, columnas=10,
        licencia="Dominio público (declarado en OpenML)",
        citacion=("Wickham, H. (2016). ggplot2: Elegant graphics for data analysis (2.ª ed.). "
                  "Springer. Conjunto publicado en OpenML con data_id 42225."),
        usos=("LAB 3",),
        nota=("Las variables `cut`, `color` y `clarity` son ordinales, no nominales. Es el mejor "
              "conjunto disponible para que se entienda la diferencia entre codificación ordinal "
              "e indicadora, y por qué elegir mal destruye información."),
        advertencia=("Se accede por OpenML y no por seaborn: el repositorio de datos de seaborn "
                     "no tiene archivo de licencia y su README advierte que los conjuntos pueden "
                     "cambiar o eliminarse sin aviso."),
    ),

    "bosque": Conjunto(
        clave="bosque",
        nombre="Covertype — tipo de cobertura forestal a partir de variables cartográficas",
        fuente="openml", ref=1596, objetivo="class", tarea="clasificación",
        filas=581012, columnas=55,
        licencia="CC BY 4.0 (fuente original: UCI, id 31)",
        citacion=("Blackard, J. (1998). Covertype [Conjunto de datos]. UCI Machine Learning "
                  "Repository. https://doi.org/10.24432/C50K5N"),
        usos=("LAB 11 (alternativa)",),
        nota=("Más de medio millón de filas. Se usa para que el costo computacional de los "
              "métodos de ensamble se note de verdad y para introducir la potenciación basada "
              "en histogramas frente a la clásica."),
        advertencia="Descarga grande (unos 75 MB). Prever el tiempo en clase.",
    ),

    "agro_ar": Conjunto(
        clave="agro_ar",
        nombre="Estimaciones Agrícolas — producción y rendimiento por cultivo, provincia y campaña",
        fuente="url",
        ref=("https://datos.magyp.gob.ar/dataset/9e1e77ba-267e-4eaa-a59f-3296e86b5f36/resource/"
             "95d066e6-8a0f-4a80-b59d-6f28f88eacd5/download/estimaciones-agricolas-2026-03.csv"),
        objetivo="rendimiento_kgxha", tarea="regresión",
        filas=None, columnas=11,
        licencia="CC BY 4.0",
        citacion=("Ministerio de Agricultura, Ganadería y Pesca de la Nación. (2026). "
                  "Estimaciones agrícolas [Conjunto de datos]. datos.magyp.gob.ar"),
        usos=("LAB 6", "Unidad 4 — tablero"),
        nota=("Conjunto argentino con cuatro dimensiones categóricas cruzables (cultivo, "
              "campaña, provincia y departamento) desde la campaña 1969/1970. Es el conjunto de "
              "referencia para el tablero, y el puente natural hacia la versión de la asignatura "
              "para Agroinformática."),
        lectura={"sep": ";", "encoding": "utf-8", "decimal": ","},
        advertencia=("La ficha del portal declara «última actualización 2019» y el archivo es de "
                     "2026: la metadata del portal no es confiable, hay que abrir el archivo. "
                     "Verificar el separador y el decimal en la primera lectura del cuatrimestre "
                     "y ajustar `lectura` en este catálogo si cambiaron."),
    ),

    "sube": Conjunto(
        clave="sube",
        nombre="SUBE — usuarios de transporte público por día en el AMBA",
        fuente="url",
        ref="https://archivos-datos.transporte.gob.ar/upload/Sube/total-usuarios-por-dia-AMBA.csv",
        objetivo="total_amba", tarea="regresión",
        filas=None, columnas=5,
        licencia="CC BY 4.0",
        citacion=("Ministerio de Transporte de la Nación. (2024). SUBE — total de usuarios por "
                  "día, AMBA [Conjunto de datos]. archivos-datos.transporte.gob.ar"),
        usos=("LAB 6 (alternativa)", "Unidad 2 — series"),
        nota=("Serie diaria desde 2020 con el corte de la pandemia perfectamente visible. Sirve "
              "para discutir por qué la validación cruzada aleatoria no corresponde en datos con "
              "estructura temporal."),
        advertencia=("La serie termina en febrero de 2024 aunque el portal declare actualización "
                     "diaria. Verificar la última fecha antes de usarlo en clase."),
    ),
}


# ------------------------------------------------------------------------- ubicaciones
def ruta_cache() -> Path:
    """Directorio de caché. Se resuelve solo según el entorno de ejecución."""
    env = os.environ.get("DATOS_CATEDRA")
    if env:
        p = Path(env)
    elif Path("/content").exists():                      # servicio de cuadernos en la nube
        p = Path("/content/datos_catedra")
    else:                                                # local o contenedor
        p = Path(__file__).resolve().parent / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ruta_manifiesto() -> Path:
    return Path(__file__).resolve().parent / "manifiesto.json"


# ----------------------------------------------------------------------------- huella
def huella(df: pd.DataFrame) -> str:
    """Huella estable de un DataFrame: forma, nombres de columna y tipos.

    No usa el contenido, de modo que es reproducible entre versiones de pandas y de
    formatos de archivo. Alcanza para detectar que la fuente cambió de esquema o de
    tamaño, que es el modo de falla que interesa prevenir.
    """
    canon = "|".join([
        f"filas={len(df)}",
        f"columnas={df.shape[1]}",
        "nombres=" + ",".join(map(str, df.columns)),
        "tipos=" + ",".join(str(t) for t in df.dtypes),
    ])
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def _leer_manifiesto() -> dict:
    p = _ruta_manifiesto()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"version": VERSION_CATALOGO, "conjuntos": {}}


def _escribir_manifiesto(m: dict) -> None:
    _ruta_manifiesto().write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- descarga
def modo_sin_conexion() -> bool:
    """True si está activo el modo sin conexión (variable DATOS_CATEDRA_OFFLINE)."""
    return os.environ.get("DATOS_CATEDRA_OFFLINE", "").strip() in ("1", "true", "True", "si", "sí")


def _descargar(c: Conjunto) -> pd.DataFrame:
    if modo_sin_conexion():
        from datos.sinteticos import generar
        return generar(c.clave)

    if c.fuente == "openml":
        from sklearn.datasets import fetch_openml
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = fetch_openml(data_id=int(c.ref), as_frame=True, parser="auto")
        return d.frame

    if c.fuente == "uci":
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as e:                                    # pragma: no cover
            raise ImportError(
                "Falta el paquete `ucimlrepo`. Instalar con: pip install ucimlrepo"
            ) from e
        d = fetch_ucirepo(id=int(c.ref))
        X = d.data.features
        y = d.data.targets
        if y is not None and len(getattr(y, "columns", [])) > 0:
            return pd.concat([X, y], axis=1)
        return X

    if c.fuente == "url":
        return pd.read_csv(str(c.ref), **c.lectura)

    raise ValueError(f"Fuente desconocida: {c.fuente}")


# ------------------------------------------------------------------------------ carga
def cargar(clave: str, *, xy: bool = False, forzar: bool = False,
           verificar: bool = True, silencioso: bool = False):
    """Carga un conjunto del catálogo, usando la caché local si existe.

    Parámetros
    ----------
    clave : str
        Clave del catálogo. `listar()` muestra las disponibles.
    xy : bool
        Si es True devuelve la tupla (X, y) en lugar del DataFrame completo.
    forzar : bool
        Vuelve a descargar aunque exista la caché.
    verificar : bool
        Compara la huella contra el manifiesto de la cátedra y avisa si difiere.
    """
    if clave not in CATALOGO:
        disponibles = ", ".join(sorted(CATALOGO))
        raise KeyError(f"«{clave}» no está en el catálogo. Disponibles: {disponibles}")

    c = CATALOGO[clave]
    sufijo = ".offline.parquet" if modo_sin_conexion() else ".parquet"
    destino = ruta_cache() / f"{clave}{sufijo}"

    if destino.exists() and not forzar:
        df = pd.read_parquet(destino)
        origen = "caché local"
    else:
        if not silencioso:
            print(f"Descargando «{clave}» desde {c.fuente}… (solo la primera vez)")
        df = _descargar(c)
        # LA CACHÉ SE ESCRIBE Y SE VUELVE A LEER. No es una vuelta al pedo: el marco
        # que devuelve la descarga y el que sale del parquet NO siempre tienen los
        # mismos tipos, y la huella se calcula sobre los tipos. Sin esta relectura,
        # quien descarga registra una huella y quien lee de la caché obtiene otra, de
        # modo que el control de integridad avisa de un cambio de fuente que no
        # existe. Pasó con los dos conjuntos que vienen de UCI, `aire` y `porotos`, y
        # no con los tres de OpenML. Con la relectura, todos calculan lo mismo.
        #
        # Y SE COMPRUEBA EL TAMAÑO, no sólo que el archivo aparezca: cuando falta el
        # motor de parquet, la escritura alcanza a crear un archivo de cero bytes y
        # recién después falla. Ese archivo vacío aparentaba una caché válida.
        try:
            df.to_parquet(destino, index=False)
            if destino.stat().st_size == 0:
                raise OSError("el archivo quedó vacío")
            df = pd.read_parquet(destino)
        except Exception as e:                                       # pragma: no cover
            if destino.exists() and destino.stat().st_size == 0:
                destino.unlink()
            warnings.warn(
                f"No se pudo escribir la caché de «{clave}»: {e}\n"
                f"  El conjunto se devuelve igual, pero NO quedó guardado: la próxima "
                f"corrida lo vuelve a descargar.\n"
                f"  Si el mensaje habla de pyarrow o fastparquet, falta el motor de "
                f"parquet en este intérprete: pip install pyarrow")
        origen = c.fuente

    # --- verificación de integridad
    h = huella(df)
    if verificar and not modo_sin_conexion():
        m = _leer_manifiesto().get("conjuntos", {}).get(clave)
        if m is None:
            m2 = _leer_manifiesto()
            m2.setdefault("conjuntos", {})[clave] = {
                "huella": h, "filas": len(df), "columnas": df.shape[1],
                "registrado_en_primera_carga": True,
            }
            _escribir_manifiesto(m2)
            if not silencioso:
                print(f"  Huella registrada por primera vez: {h}")
        elif m.get("huella") != h:
            warnings.warn(
                f"\nLa huella de «{clave}» NO coincide con el manifiesto de la cátedra.\n"
                f"  esperada: {m.get('huella')}  ({m.get('filas')} filas, "
                f"{m.get('columnas')} columnas)\n"
                f"  obtenida: {h}  ({len(df)} filas, {df.shape[1]} columnas)\n"
                f"La fuente cambió. Avisar a la cátedra antes de continuar: los resultados "
                f"no serán comparables con los del resto de la comisión.",
                stacklevel=2,
            )

    if not silencioso:
        if modo_sin_conexion():
            print("!" * 78)
            print("MODO SIN CONEXIÓN: estos NO son los datos reales, sino un sustituto")
            print("sintético con la misma estructura. Sirve para dictar y para probar el")
            print("código, pero NINGÚN resultado obtenido así es válido para una entrega.")
            print("!" * 78)
        print(f"«{clave}»: {len(df):,} filas × {df.shape[1]} columnas  (origen: {origen})"
              .replace(",", "."))
        if c.advertencia:
            print(f"  AVISO: {c.advertencia}")

    if xy:
        if c.objetivo is None:
            raise ValueError(f"«{clave}» no tiene variable objetivo definida (tarea: {c.tarea}).")
        if c.objetivo not in df.columns:
            raise KeyError(
                f"La columna objetivo «{c.objetivo}» no está en el conjunto descargado. "
                f"Columnas disponibles: {list(df.columns)[:15]}…")
        return df.drop(columns=[c.objetivo]), df[c.objetivo]
    return df


# ------------------------------------------------------------------------- exploración
def listar(uso: str | None = None, tarea: str | None = None) -> pd.DataFrame:
    """Devuelve el catálogo como tabla. Se puede filtrar por uso o por tipo de tarea."""
    filas = []
    for c in CATALOGO.values():
        if uso and not any(uso.lower() in u.lower() for u in c.usos):
            continue
        if tarea and tarea.lower() not in c.tarea.lower():
            continue
        filas.append({
            "clave": c.clave,
            "nombre": c.nombre.split(" — ")[0],
            "tarea": c.tarea,
            "filas": c.filas,
            "columnas": c.columnas,
            "objetivo": c.objetivo or "—",
            "fuente": f"{c.fuente}:{c.ref}" if c.fuente != "url" else "url",
            "usos": ", ".join(c.usos),
        })
    return pd.DataFrame(filas)


def ficha(clave: str) -> None:
    """Imprime la ficha completa de un conjunto: contexto, licencia y citación."""
    if clave not in CATALOGO:
        raise KeyError(f"«{clave}» no está en el catálogo.")
    c = CATALOGO[clave]
    ancho = 78
    print("=" * ancho)
    print(c.nombre)
    print("=" * ancho)
    print(f"Clave          : {c.clave}")
    print(f"Tarea          : {c.tarea}")
    print(f"Objetivo       : {c.objetivo or '— (sin variable objetivo)'}")
    print(f"Tamaño         : {c.filas or '?'} filas × {c.columnas or '?'} columnas")
    print(f"Fuente         : {c.fuente} → {c.ref}")
    print(f"Licencia       : {c.licencia}")
    print(f"Se usa en      : {', '.join(c.usos)}")
    print("-" * ancho)
    print("Por qué este conjunto:")
    for linea in _envolver(c.nota, ancho - 2):
        print(f"  {linea}")
    if c.advertencia:
        print("-" * ancho)
        print("Advertencia:")
        for linea in _envolver(c.advertencia, ancho - 2):
            print(f"  {linea}")
    print("-" * ancho)
    print("Citación (APA 7):")
    for linea in _envolver(c.citacion, ancho - 2):
        print(f"  {linea}")
    print("=" * ancho)


def _envolver(texto: str, ancho: int) -> list[str]:
    palabras, lineas, actual = texto.split(), [], ""
    for p in palabras:
        if len(actual) + len(p) + 1 > ancho:
            lineas.append(actual)
            actual = p
        else:
            actual = f"{actual} {p}".strip()
    if actual:
        lineas.append(actual)
    return lineas


def verificar_todos(silencioso: bool = False) -> pd.DataFrame:
    """Descarga o carga todos los conjuntos y reporta el estado de cada uno.

    Pensado para que la cátedra lo ejecute una vez antes de que empiece el cuatrimestre.
    """
    filas = []
    for clave in CATALOGO:
        try:
            df = cargar(clave, verificar=True, silencioso=True)
            filas.append({"clave": clave, "estado": "OK", "filas": len(df),
                          "columnas": df.shape[1], "huella": huella(df), "detalle": ""})
        except Exception as e:
            filas.append({"clave": clave, "estado": "ERROR", "filas": None,
                          "columnas": None, "huella": None,
                          "detalle": f"{type(e).__name__}: {e}"})
        if not silencioso:
            print(f"  {filas[-1]['estado']:6} {clave}")
    return pd.DataFrame(filas)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ficha":
        ficha(sys.argv[2])
    else:
        print(listar().to_string(index=False))
