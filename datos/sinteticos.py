"""
Modo sin conexión — sustitutos sintéticos del catálogo
======================================================
Aprendizaje Automático y Grandes Datos — UNRaf

Para qué existe
---------------
1. **Respaldo en el laboratorio.** Si la red de la facultad falla el día de la práctica,
   la clase se dicta igual: los cuadernos corren completos contra estos sustitutos.
2. **Verificación de los cuadernos.** Permite ejecutar todas las prácticas de punta a
   punta sin descargar nada, para comprobar que no hay errores antes de dictarlas.

Qué garantizan y qué no
-----------------------
Cada sustituto preserva la ESTRUCTURA del problema original: tipo de tarea, presencia de
variables categóricas, valores faltantes, desbalance de clases y forma de la relación.
NO preserva los valores reales ni las conclusiones sustantivas.

Los resultados obtenidos en modo sin conexión NO son válidos para una entrega. La carga
imprime un aviso destacado para que nadie confunda una cosa con la otra.

Activación
----------
    export DATOS_CATEDRA_OFFLINE=1      # Linux y macOS
    set DATOS_CATEDRA_OFFLINE=1         # Windows

o desde Python, antes de importar el catálogo:

    import os; os.environ["DATOS_CATEDRA_OFFLINE"] = "1"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEMILLA = 20260724


def _rng(clave: str) -> np.random.Generator:
    """Generador propio por conjunto: cada clave da siempre los mismos datos."""
    return np.random.default_rng(SEMILLA + (abs(hash(clave)) % 10_000))


# ---------------------------------------------------------------------- generadores
def _bici(n=4000):
    """Sustituto de `bici`, con el ESQUEMA EXACTO del conjunto real.

    La versión anterior de esta función estaba escrita mirando el archivo horario
    original de UCI —con `hr`, `hum`, `weathersit`, `casual` y `registered`— y no la
    versión de OpenML que el catálogo descarga, que renombra las columnas y no trae
    las dos últimas. La consecuencia fue que cuatro laboratorios nombraban columnas
    que sólo existían acá, y el LAB 5 construía todo un contraejemplo de fuga sobre
    `casual` y `registered`: corría perfecto sin conexión y se caía con datos reales,
    que es como se dicta.
    """
    r = _rng("bici")
    hora = r.integers(0, 24, n)
    estacion = r.integers(0, 4, n)
    laborable = r.integers(0, 2, n)
    clima = r.choice([0, 1, 2, 3], n, p=[0.65, 0.25, 0.09, 0.01])
    temp = np.clip(10 + 12 * np.sin(estacion * np.pi / 2) + r.normal(0, 3, n), 0.82, 41.0)
    humedad = np.clip(r.normal(0.60, 0.15, n), 0.0, 1.0)
    viento = np.clip(r.gamma(2, 4, n), 0, 56.9969)
    sensacion = np.clip(temp + 4 * (1 - humedad) - 0.2 * viento + r.normal(0, 1.5, n), 0.0, 50.0)
    # demanda con doble pico en días laborables y pico único los fines de semana
    base = np.where(
        laborable == 1,
        90 * np.exp(-((hora - 8) ** 2) / 6) + 110 * np.exp(-((hora - 18) ** 2) / 6),
        70 * np.exp(-((hora - 14) ** 2) / 20),
    )
    efecto = base * (1 + 0.03 * (temp - 15)) * (1 - 0.4 * (humedad - 0.5)) / ((clima + 1) ** 0.5)
    total = np.clip(efecto + r.normal(0, 12, n), 1, 977).round().astype(int)

    ESTACIONES = ["spring", "summer", "fall", "winter"]
    CLIMAS = ["clear", "misty", "rain", "heavy_rain"]
    # Las filas del conjunto real vienen en orden cronológico: dos años de horas
    # consecutivas. Se reproduce, porque de ese orden depende el ejercicio de fuga
    # temporal y una partición aleatoria tiene que poder equivocarse igual que allá.
    anio = np.sort(r.integers(0, 2, n))
    mes = r.integers(1, 13, n)
    orden = np.lexsort((hora, mes, anio))
    def o(v):
        return np.asarray(v)[orden]

    return pd.DataFrame({
        "season": pd.Categorical([ESTACIONES[i] for i in o(estacion)], categories=ESTACIONES),
        "year": o(anio),
        "month": o(mes),
        "hour": o(hora),
        "holiday": pd.Categorical([str(bool(v)) for v in o(r.binomial(1, 0.03, n))],
                                  categories=["False", "True"]),
        "weekday": o(r.integers(0, 7, n)),
        "workingday": pd.Categorical([str(bool(v)) for v in o(laborable)],
                                     categories=["False", "True"]),
        "weather": pd.Categorical([CLIMAS[i] for i in o(clima)], categories=CLIMAS),
        "temp": o(temp).round(2),
        "feel_temp": o(sensacion).round(3),
        "humidity": o(humedad).round(2),
        "windspeed": o(viento).round(4),
        "count": o(total),
    })


def _mixto_desbalanceado(clave, n, p_pos, n_num, n_cat, objetivo, faltantes=0.0,
                         marca_faltante=None, etiquetas=("no", "yes"),
                         orden=None, nombres_num=None, nombres_cat=None,
                         posterior=None):
    """Constructor genérico de un problema binario desbalanceado con variables mixtas.

    `orden`, `nombres_num` y `nombres_cat` existen para que el sustituto tenga los
    NOMBRES REALES del conjunto que reemplaza. Sin eso, un cuaderno que nombre una
    columna corre en un modo y se cae en el otro, y la prueba de regresión sin
    conexión no lo detecta porque ahí la columna se llama distinto.

    `posterior` designa una columna numérica que se genera DESPUÉS de la respuesta y
    en función de ella: es la variable que sólo se conoce una vez ocurrido el hecho
    que se quiere predecir. Sin ella el contraejemplo de fuga del LAB 5 no tendría
    nada que mostrar en modo sin conexión.
    """
    r = _rng(clave)
    nom_n = list(nombres_num) if nombres_num else [f"num_{i}" for i in range(n_num)]
    nom_c = list(nombres_cat) if nombres_cat else [f"cat_{i}" for i in range(n_cat)]
    if len(nom_n) != n_num or len(nom_c) != n_cat:
        raise ValueError(f"«{clave}»: se pidieron {n_num} numéricas y {n_cat} categóricas, "
                         f"y llegaron {len(nom_n)} y {len(nom_c)} nombres.")
    num = {nom: r.normal(0, 1, n) for nom in nom_n}
    cat = {nom: r.choice([f"n{j}" for j in range(r.integers(3, 7))], n) for nom in nom_c}
    logit = sum(num[nom] * (0.6 / (i + 1)) for i, nom in enumerate(nom_n)
                if nom != posterior)
    logit += np.where(list(cat.values())[0] == "n0", 0.9, -0.2)
    logit += np.log(p_pos / (1 - p_pos)) - logit.mean()
    y = r.binomial(1, 1 / (1 + np.exp(-logit)))

    if posterior is not None:
        # Se genera a partir de `y`, que es exactamente lo que la vuelve inservible
        # para predecir: para conocer su valor hay que esperar a que el hecho ocurra.
        num[posterior] = np.clip(
            r.lognormal(np.where(y == 1, 6.0, 5.0), 0.75, n), 0, 4918).round().astype(int)

    df = pd.DataFrame({**num, **cat})
    if faltantes > 0:
        for col in df.columns:
            mask = r.random(n) < faltantes
            df.loc[mask, col] = marca_faltante if marca_faltante is not None else np.nan
    df[objetivo] = pd.Categorical([etiquetas[v] for v in y], categories=list(etiquetas))
    if orden:
        faltan = set(df.columns) ^ set(orden)
        if faltan:
            raise ValueError(f"«{clave}»: el orden declarado no cubre {sorted(faltan)}.")
        df = df[list(orden)]
    return df


def _aire(n=3000):
    """Sustituto de `aire`, con las 15 columnas del conjunto real.

    La versión anterior generaba 9 y omitía `Date`, `Time` y cuatro sensores. Un
    cuaderno que nombrara cualquiera de esas seis corría con datos reales y se caía
    sin conexión, que es el mismo defecto al revés.
    """
    r = _rng("aire")
    t = np.arange(n)
    base = 8 + 4 * np.sin(2 * np.pi * t / 24) + 2 * np.sin(2 * np.pi * t / (24 * 7))
    inicio = pd.Timestamp("2004-03-10 18:00:00")
    sello = inicio + pd.to_timedelta(t, unit="h")
    df = pd.DataFrame({
        # El real trae fecha y hora como TEXTO, en dos columnas separadas y con el
        # formato de origen. Es parte del ejercicio de lectura y por eso se reproduce.
        "Date": sello.strftime("%-d/%-m/%Y"),
        "Time": sello.strftime("%H:%M:%S"),
        "CO(GT)": (base * 0.3 + r.normal(0, 0.4, n)).round(1),
        "PT08.S1(CO)": (1000 + base * 40 + r.normal(0, 60, n)).round(0).astype(int),
        "NMHC(GT)": (base * 12 + r.normal(0, 20, n)).round(0).astype(int),
        "C6H6(GT)": (base + r.normal(0, 1.2, n)).round(1),
        "PT08.S2(NMHC)": (900 + base * 35 + r.normal(0, 50, n)).round(0).astype(int),
        "NOx(GT)": (base * 25 + r.normal(0, 40, n)).round(0).astype(int),
        "PT08.S3(NOx)": (1100 - base * 30 + r.normal(0, 55, n)).round(0).astype(int),
        "NO2(GT)": (base * 11 + r.normal(0, 18, n)).round(0).astype(int),
        "PT08.S4(NO2)": (1500 + base * 45 + r.normal(0, 70, n)).round(0).astype(int),
        "PT08.S5(O3)": (950 + base * 38 + r.normal(0, 65, n)).round(0).astype(int),
        "T": (18 + 8 * np.sin(2 * np.pi * t / (24 * 365)) + r.normal(0, 2, n)).round(1),
        "RH": np.clip(r.normal(55, 15, n), 5, 100).round(1),
        "AH": np.clip(r.normal(1.0, 0.35, n), 0.1, None).round(4),
    })
    # LA TRAMPA DEL CONJUNTO ORIGINAL: faltantes codificados como -200, no como nulos.
    for col in df.columns:
        if col in ("Date", "Time"):
            continue
        df.loc[r.random(n) < 0.09, col] = -200
    return df


def _porotos(n=3500):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=16, n_informative=8, n_redundant=5,
                               n_classes=7, n_clusters_per_class=1, class_sep=1.3,
                               weights=[0.26, 0.19, 0.15, 0.14, 0.11, 0.09, 0.06],
                               random_state=SEMILLA)
    nombres = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRatio",
               "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
               "Roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
               "ShapeFactor3", "ShapeFactor4"]
    variedades = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA"]
    df = pd.DataFrame(X, columns=nombres)
    df["Class"] = [variedades[i] for i in y]      # texto, como en el conjunto real
    return df


def _retail(n_facturas=6000):
    r = _rng("retail")
    productos = [f"{r.integers(10000, 99999)}" for _ in range(220)]
    descripciones = {p: f"PRODUCTO {i:03d}" for i, p in enumerate(productos)}
    # familias de coocurrencia: hacen que existan reglas de asociación reales
    familias = [productos[i:i + 6] for i in range(0, 180, 6)]
    filas = []
    for f in range(n_facturas):
        fam = familias[r.integers(0, len(familias))]
        k = int(np.clip(r.poisson(4) + 1, 1, 12))
        items = list(r.choice(fam, size=min(k, len(fam)), replace=False))
        if r.random() < 0.35:
            items += list(r.choice(productos, size=r.integers(1, 4), replace=False))
        cancel = r.random() < 0.02
        for it in set(items):
            filas.append({
                "Invoice": ("C" if cancel else "") + str(500000 + f),
                "StockCode": it,
                # descripción inconsistente en el 3 % de los casos: parte del ejercicio
                "Description": descripciones[it] if r.random() > 0.03 else descripciones[it].lower(),
                "Quantity": int(-r.integers(1, 5) if cancel else r.integers(1, 25)),
                # El real trae la fecha como TEXTO, no como marca temporal: convertirla es
                # parte del ejercicio de lectura del LAB 2.
                "InvoiceDate": (pd.Timestamp("2010-01-01")
                                + pd.Timedelta(days=int(r.integers(0, 700)))
                                + pd.Timedelta(minutes=int(r.integers(0, 1440)))
                                ).strftime("%Y-%m-%d %H:%M:%S"),
                "Price": round(float(r.gamma(2, 2)), 2),
                "Customer_ID": float(r.integers(12000, 18000)) if r.random() > 0.22 else np.nan,
                "Country": r.choice(["United Kingdom", "France", "Germany", "EIRE", "Spain"],
                                    p=[0.85, 0.05, 0.04, 0.03, 0.03]),
            })
    return pd.DataFrame(filas)


def _actividad(n=2400):
    """Estructura no esférica: grupos alargados y curvos, como el conjunto original.

    LOS NOMBRES Y LAS ETIQUETAS SON LOS DEL CONJUNTO REAL, que llegó a la caché el
    1 de agosto de 2026. Antes generaba `f000` a `f056` y etiquetas con el nombre de
    la actividad; el real trae `V1` a `V561` y la etiqueta como número del 1 al 6. Un
    cuaderno que nombre una columna corre en un modo y se cae en el otro, que es el
    defecto que ya apareció con otros siete conjuntos del catálogo.
    """
    r = _rng("actividad")
    actividades = [str(i) for i in range(1, 7)]
    bloques, etiquetas = [], []
    for i, act in enumerate(actividades):
        m = n // 6
        t = np.linspace(0, 3 * np.pi, m)
        # las dinámicas describen trayectorias curvas; las estáticas, nubes solapadas
        if i < 3:
            centro = np.column_stack([np.cos(t + i) * (3 + i), np.sin(t + i) * (3 + i), t])
        else:
            centro = np.column_stack([np.full(m, -6 + i), np.full(m, 4.0), np.full(m, i * 0.4)])
        base = centro + r.normal(0, 0.55 if i < 3 else 1.1, centro.shape)
        ruido = r.normal(0, 1, (m, 561))
        proj = r.normal(0, 1, (3, 561))
        bloques.append(base @ proj * 1.6 + ruido)
        etiquetas += [act] * m
    X = np.vstack(bloques)
    df = pd.DataFrame(X, columns=[f"V{i + 1}" for i in range(X.shape[1])])
    df["Activity"] = pd.Categorical(etiquetas, categories=actividades)
    return df


def _adultos(n=5000):
    r = _rng("adultos")
    edad = np.clip(r.normal(38, 13, n), 17, 90).astype(int)
    educ = r.integers(1, 17, n)
    horas = np.clip(r.normal(40, 12, n), 1, 99).astype(int)
    sexo = r.choice(["Male", "Female"], n, p=[0.67, 0.33])
    raza = r.choice(["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                    n, p=[0.85, 0.10, 0.03, 0.01, 0.01])
    trabajo = r.choice(["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"],
                       n, p=[0.70, 0.11, 0.08, 0.06, 0.05]).astype(object)
    ocup = r.choice(["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Prof-specialty",
                     "Other-service", "Machine-op-inspct"], n).astype(object)
    pais = r.choice(["United-States", "Mexico", "Philippines", "Germany"],
                    n, p=[0.90, 0.05, 0.03, 0.02]).astype(object)
    logit = (0.05 * (edad - 38) + 0.28 * (educ - 9) + 0.035 * (horas - 40)
             + np.where(sexo == "Male", 0.75, 0.0) - 2.0)
    y = r.binomial(1, 1 / (1 + np.exp(-logit)))
    # LA TRAMPA DEL CONJUNTO ORIGINAL: faltantes codificados como "?", no como nulos.
    for arr in (trabajo, ocup, pais):
        arr[r.random(n) < 0.06] = "?"
    return pd.DataFrame({
        "age": edad, "workclass": trabajo, "education-num": educ, "occupation": ocup,
        "race": raza, "sex": sexo, "hours-per-week": horas, "native-country": pais,
        "class": pd.Categorical([">50K" if v else "<=50K" for v in y],
                                categories=["<=50K", ">50K"]),
    })


def _diamantes(n=5000):
    r = _rng("diamantes")
    corte = pd.Categorical(r.choice(["Fair", "Good", "Very Good", "Premium", "Ideal"], n,
                                    p=[0.03, 0.09, 0.22, 0.26, 0.40]),
                           categories=["Fair", "Good", "Very Good", "Premium", "Ideal"],
                           ordered=True)
    color = pd.Categorical(r.choice(list("JIHGFED"), n), categories=list("JIHGFED"), ordered=True)
    clar = pd.Categorical(r.choice(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], n),
                          categories=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
                          ordered=True)
    quilates = np.clip(r.gamma(2.2, 0.35, n), 0.2, 5.0)
    precio = (np.exp(7.0 + 1.9 * np.log(quilates)
                     + 0.06 * corte.codes + 0.07 * color.codes + 0.08 * clar.codes
                     + r.normal(0, 0.16, n))).round().astype(int)
    return pd.DataFrame({
        "carat": quilates.round(2), "cut": corte, "color": color, "clarity": clar,
        "depth": (61.7 + r.normal(0, 1.4, n)).round(1),
        "table": (57.5 + r.normal(0, 2.2, n)).round(1),
        "x": (quilates ** (1 / 3) * 5.6).round(2),
        "y": (quilates ** (1 / 3) * 5.6 + r.normal(0, 0.05, n)).round(2),
        "z": (quilates ** (1 / 3) * 3.4).round(2),
        "price": precio,
    })


def _agro_ar(n_dep=90):
    r = _rng("agro_ar")
    cultivos = ["Soja total", "Maíz", "Trigo total", "Girasol", "Sorgo"]
    provs = {"Santa Fe": ["Castellanos", "Las Colonias", "San Martín", "Belgrano"],
             "Córdoba": ["Río Cuarto", "Marcos Juárez", "Unión"],
             "Buenos Aires": ["Pergamino", "Junín", "Nueve de Julio", "Tandil"],
             "Entre Ríos": ["Paraná", "Gualeguaychú"]}
    filas = []
    for anio in range(1990, 2026):
        for cult in cultivos:
            for prov, deps in provs.items():
                for dep in deps:
                    sup = float(np.clip(r.gamma(3, 9000), 500, None))
                    perdida = r.beta(1.2, 18)
                    rend = float(np.clip(r.normal(2600 + 42 * (anio - 1990), 750), 300, None))
                    cos = sup * (1 - perdida)
                    filas.append({
                        "cultivo": cult, "anio": anio,
                        "campania": f"{anio}/{str(anio + 1)[2:]}",
                        "provincia": prov, "departamento": dep,
                        "superficie_sembrada_ha": round(sup, 1),
                        "superficie_cosechada_ha": round(cos, 1),
                        "produccion_tm": round(cos * rend / 1000, 1),
                        "rendimiento_kgxha": round(rend, 1),
                    })
    df = pd.DataFrame(filas)
    df.loc[_rng("agro_ar2").random(len(df)) < 0.04, "rendimiento_kgxha"] = np.nan
    return df


def _sube():
    r = _rng("sube")
    fechas = pd.date_range("2020-01-01", "2024-02-16", freq="D")
    n = len(fechas)
    finde = fechas.dayofweek >= 5
    base = np.where(finde, 2.1e6, 5.4e6)
    # caída de la pandemia y recuperación gradual
    dias = (fechas - fechas[0]).days.to_numpy()
    pandemia = np.where((dias > 79) & (dias < 200), 0.12,
                        np.where(dias < 79, 1.0, np.clip(0.12 + (dias - 200) / 900, 0, 1)))
    total = (base * pandemia * (1 + r.normal(0, 0.05, n))).clip(1e4)
    return pd.DataFrame({
        # EL REAL TRAE LA FECHA COMO TEXTO, no como marca temporal. El sustituto la
        # entregaba ya convertida y por eso el laboratorio de flujos corría con el
        # sintético y se rompía con el conjunto real el 2 de agosto de 2026: `.dt`
        # no existe sobre una columna de texto. Un sustituto más fácil que el real
        # es peor que no tener sustituto, porque da una corrida verde que no vale.
        "indice_tiempo": fechas.strftime("%Y-%m-%d"),
        "total_amba": total.round(0),
        "colectivo_amba": (total * 0.62).round(0),
        "subte_amba": (total * 0.14).round(0),
        "tren_amba": (total * 0.24).round(0),
    })


def _bosque(n=20000):
    """Sustituto de `bosque`, con los nombres y la cantidad de columnas del real.

    El real tiene 10 variables continuas, 4 indicadoras de área silvestre y 40 de tipo
    de suelo. La versión anterior generaba `var_0` a `var_11` y `tipo_suelo_0` a
    `tipo_suelo_3`, que no son los nombres de ninguna columna real.
    """
    from sklearn.datasets import make_classification
    CONTINUAS = ["Elevation", "Aspect", "Slope",
                 "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
                 "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
                 "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
    X, y = make_classification(n_samples=n, n_features=len(CONTINUAS), n_informative=8,
                               n_redundant=2, n_classes=7, n_clusters_per_class=1,
                               class_sep=0.9,
                               weights=[0.49, 0.36, 0.06, 0.03, 0.02, 0.02, 0.02],
                               random_state=SEMILLA)
    r = _rng("bosque")
    df = pd.DataFrame(X, columns=CONTINUAS)
    # Las indicadoras del real son excluyentes dentro de cada familia: una sola en 1.
    area = r.integers(0, 4, n)
    for i in range(4):
        df[f"Wilderness_Area{i + 1}"] = pd.Categorical(
            (area == i).astype(int).astype(str), categories=["0", "1"])
    suelo = r.integers(0, 40, n)
    for i in range(40):
        df[f"Soil_Type{i + 1}"] = pd.Categorical(
            (suelo == i).astype(int).astype(str), categories=["0", "1"])
    df["class"] = pd.Categorical(y.astype(str))
    return df


GENERADORES = {
    "bici": _bici,
    # Los nombres y el orden son los del conjunto real. `banco` viene anonimizado como
    # V1 a V16 y `V12` es la duración de la llamada, que se conoce recién cuando la
    # llamada terminó: por eso se genera como `posterior`, a partir de la respuesta.
    "banco": lambda: _mixto_desbalanceado(
        "banco", 6000, 0.117, 7, 9, "Class",
        nombres_num=["V1", "V6", "V10", "V12", "V13", "V14", "V15"],
        nombres_cat=["V2", "V3", "V4", "V5", "V7", "V8", "V9", "V11", "V16"],
        posterior="V12",
        orden=[f"V{i}" for i in range(1, 17)] + ["Class"],
        etiquetas=("1", "2")),
    "autos": lambda: _mixto_desbalanceado(
        "autos", 8000, 0.123, 14, 18, "IsBadBuy", faltantes=0.13, etiquetas=("0", "1"),
        nombres_num=["PurchDate", "VehYear", "VehicleAge", "VehOdo",
                     "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice",
                     "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice",
                     "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice",
                     "MMRCurrentRetailAveragePrice", "MMRCurrentRetailCleanPrice",
                     "VehBCost", "WarrantyCost"],
        nombres_cat=["Auction", "Make", "Model", "Trim", "SubModel", "Color",
                     "Transmission", "WheelTypeID", "WheelType", "Nationality", "Size",
                     "TopThreeAmericanName", "PRIMEUNIT", "AUCGUART", "BYRNO", "VNZIP1",
                     "VNST", "IsOnlineSale"],
        orden=["IsBadBuy", "PurchDate", "Auction", "VehYear", "VehicleAge", "Make", "Model", "Trim",
               "SubModel", "Color", "Transmission", "WheelTypeID", "WheelType", "VehOdo",
               "Nationality", "Size", "TopThreeAmericanName",
               "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice",
               "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice",
               "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice",
               "MMRCurrentRetailAveragePrice", "MMRCurrentRetailCleanPrice",
               "PRIMEUNIT", "AUCGUART", "BYRNO", "VNZIP1", "VNST", "VehBCost",
               "IsOnlineSale", "WarrantyCost"]),
    "aire": _aire,
    "porotos": _porotos,
    "retail": _retail,
    "actividad": _actividad,
    "adultos": _adultos,
    "diamantes": _diamantes,
    "bosque": _bosque,
    "agro_ar": _agro_ar,
    "sube": _sube,
}


def generar(clave: str) -> pd.DataFrame:
    if clave not in GENERADORES:
        raise KeyError(f"No hay sustituto sintético para «{clave}».")
    return GENERADORES[clave]()
