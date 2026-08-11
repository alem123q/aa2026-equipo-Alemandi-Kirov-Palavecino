# Bitácora del Equipo - Laboratorio 1

## 👥 Integrantes y Entornos
| Nombre | Modalidad de Entorno | Versión de Python | Sistema Operativo |
|--------|----------------------|-------------------|-------------------|
| Alemandi | Entorno virtual local (`venv`) | 3.12.1 | Windows 10/11 |
| Kirov | Entorno virtual local (`venv`) | 3.12.1 | Windows 10/11 |
| Palavecino | Entorno virtual local (`venv`) | 3.12.1 | Windows 10/11 |

## 🧪 Prueba del Entorno Limpio
- **Responsable de la prueba:** [Nombre de quien clonó el repo]
- **Fecha:** 11 de Agosto de 2026
- **Tiempo de instalación:** ~2 minutos
- **Resultado del entorno (Python y librerías):** ÉXITO (Código 0 a nivel de dependencias).
- **Resultado de la descarga de datos:** 1 Falla y 2 Alertas de fuente.

## 📋 Salida y Análisis del Verificador
El entorno de Python se replicó sin problemas. Sin embargo, al ejecutar `python scripts/verificar_entorno.py --datos` en un entorno limpio, el catálogo detectó inconsistencias con las fuentes externas originales que no son culpa del código del equipo:

1. **Alerta `agro_ar` y `sube`:** El manifiesto de la cátedra detectó *Data Drift*. La fuente pública de `agro_ar` pasó de 11 a 1 columna, y `sube` sumó 9 filas nuevas. El catálogo bloqueó la carga para evitar silenciar el cambio.
2. **Falla `actividad`:** La API de `ucimlrepo` (UCI ML Repository) devolvió un `DatasetNotFoundError` para el dataset id=240. La plataforma externa modificó o bloqueó el acceso por API a este conjunto.

**Conclusión del equipo:** El entorno de software es 100% reproducible. Las fallas al momento de la prueba se deben a la volatilidad de las fuentes de datos en la web pública, lo que justifica la regla de la cátedra de usar el catálogo con caché local y huellas criptográficas. Para las entregas, utilizaremos la caché local ya verificada o el modo sintético offline provisto por la cátedra para los datasets afectados por estas caídas externas.


💥 Desafío 2: Romper la reproducibilidad a propósito
El objetivo: Demostrar que fijar la semilla (random_state=42 o random.seed(42)) no siempre es suficiente para garantizar el mismo resultado numérico exacto.
Explicación:
Fijar la semilla (random.seed(42)) garantiza que la secuencia de números pseudoaleatorios generada sea la misma, pero no garantiza el orden en que se procesan los datos si estos están almacenados en estructuras no ordenadas como un set (conjunto) o las claves de un dict.
Desde Python 3.3, por motivos de seguridad, el intérprete aleatoriza el "hash" de los strings cada vez que se abre una terminal nueva o se reinicia el Kernel (esto se controla mediante la variable de entorno PYTHONHASHSEED). Si nuestro código itera sobre un set, el orden en que Python nos entrega los elementos será distinto en cada ejecución.
Como las operaciones matemáticas con números de punto flotante (floats) en la computadora no son estrictamente asociativas debido al redondeo de decimales (es decir, (a + b) + c no siempre es idéntico a a + (b + c) a nivel de bits), procesar y acumular valores en distinto orden altera los últimos decimales del resultado final. Por lo tanto, el mismo modelo o métrica dará un número ligeramente distinto en máquinas distintas o en ejecuciones distintas, aunque la semilla sea la misma.

Solución para el equipo:
Para garantizar reproducibilidad estricta al trabajar con textos, IDs o categorías, siempre debemos convertir los sets o las claves de los diccionarios a listas ordenadas usando sorted(list(mi_set)) antes de iterarlas. Alternativamente, al correr scripts desde la terminal, se puede fijar la variable de entorno con PYTHONHASHSEED=0 python mi_script.py.


⚠️ Desafío 4: Qué pudo fallar (Lectura de datos de una cuenta personal)
El objetivo: Identificar los riesgos de no usar un catálogo de datos con caché y huella digital. El material anterior leía de un Google Sheets personal o URLs públicas.

1. Fallo 1: El dueño borra el archivo, lo hace privado o le suspenden la cuenta.
Consecuencia: El notebook tira un error 404 Not Found o 403 Forbidden y nadie puede correr el análisis ni rendir el examen.
Qué haber hecho: Descargar el archivo una sola vez, guardarlo en un repositorio institucional (o en datos/crudos/ si la licencia y el peso lo permiten) y versionarlo con Git.
2. Fallo 2: El dueño actualiza los datos silenciosamente (Data Drift).
Consecuencia: El dueño corrige errores ortográficos, agrega filas nuevas o cambia el nombre de una columna. El código de los estudiantes se rompe a mitad de cuatrimestre o, peor, corre pero da resultados distintos sin que nadie se dé cuenta. (Ejemplo real de hoy: el dataset agro_ar pasó de 11 columnas a 1 columna en la fuente original, y sube sumó 9 filas nuevas).
Qué haber hecho: Calcular y guardar el hash criptográfico (MD5 o SHA256) del archivo original en un manifiesto. Al cargarlo, el código verifica que el hash coincida. Si el archivo en la nube cambió, el catálogo lanza un aviso antes de procesar, bloqueando la carga silenciosa de datos erróneos.
3. Fallo 3: Problemas de red, caídas de API o límites de descarga (Rate Limiting).
Consecuencia: Si Google Sheets, la UCI o el servidor limita las descargas, el notebook falla cuando el docente o los 20 estudiantes de la comisión intentan correrlo al mismo tiempo antes de la entrega. (Ejemplo real de hoy: La API de ucimlrepo devolvió DatasetNotFoundError para el dataset de "actividad" porque la plataforma externa modificó o bloqueó el acceso por API).
Qué haber hecho: Implementar un sistema de caché local (como el que usa datos.catalogo en esta materia). El código intenta descargar, pero si ya tiene una copia local con el hash correcto, lee directamente del disco sin usar internet. Para los casos donde la API externa cae definitivamente, usar el modo offline con datos sintéticos provistos por la cátedra.