# Expenses AI Classifier

Dashboard local para revisar movimientos bancarios, aplicar reglas deterministas y usar un LLM solo cuando la clasificacion no es evidente.

Empece este proyecto para probar un flujo hibrido: reglas locales para casos claros, una taxonomia jerarquica para mantener consistencia y revision humana antes de aceptar cambios generados por IA.

## Estado

Esta version publica es una copia saneada del proyecto:

- incluye codigo de aplicacion, tests y configuracion generica;
- incluye un CSV sintetico pequeno para probar el flujo;
- no incluye movimientos bancarios reales;
- no incluye credenciales, cache de LLM, logs ni exports personales.

## Que hace

- Lee movimientos con columnas `Fecha`, `Concepto`, `Movimiento` e `Importe`.
- Aplica reglas locales por texto o regex.
- Organiza categorias en un arbol editable.
- Permite clasificacion asistida con DeepSeek si configuras una API key.
- Mantiene trazabilidad de decisiones y casos pendientes de revision.
- Puede funcionar en modo consola si la UI no esta disponible.

## Decisiones tecnicas

- Python como base del procesamiento.
- Pandas para normalizacion y agrupaciones.
- Flet para la interfaz local.
- Configuracion por archivos JSON y variables de entorno.
- Separacion entre `core`, `infra`, `ui` y `support`.
- Tests sobre procesamiento, reglas, almacenamiento, DeepSeek y entrada CLI.

## Instalacion

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Uso con datos sinteticos

El repositorio incluye `data/raw/sample_expenses.csv`.

Para probar un resumen:

```powershell
copy data\raw\sample_expenses.csv data\raw\expenses.csv
python main.py --summary --group-by CategoriaLeaf
```

Para abrir la aplicacion:

```powershell
python main.py
```

## IA opcional

La clasificacion con DeepSeek esta desactivada si no defines `DEEPSEEK_API_KEY`.

```env
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

La aplicacion no envia el CSV completo al modelo. Resume la vista actual, categorias y movimientos visibles para reducir exposicion de datos.

## Tests

```powershell
python -m pytest tests/ -v
python -m flake8 src tests
python -m mypy src
```

## Limitaciones

- La demo publica usa datos ficticios.
- La integracion de descarga bancaria esta pensada para uso local y requiere navegador instalado.
- Las decisiones de IA deben revisarse antes de convertirlas en reglas permanentes.
- No es una herramienta financiera certificada; es una aplicacion personal para analisis y categorizacion.

## Mi aportacion

Defini el problema, la taxonomia, el flujo de revision y las reglas de privacidad. Uso herramientas de programacion asistida por IA para acelerar implementacion y revisar alternativas, pero mantengo la responsabilidad sobre requisitos, pruebas y decisiones tecnicas.
