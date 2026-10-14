# Proyecto Integrado 3 — EA1: Carga y creación de dataset

Flujo de la actividad:

```
HFDatasetLoader (src/hf_dataset_loader.py) → DataFrame → transformaciones (notebook)
                                                      → DatasetSummarizer (src/dataset_summarizer.py) → output/resumen_iris.txt
```

## Estructura

| Archivo | Descripción |
|---|---|
| `src/hf_dataset_loader.py` | Clase que se conecta a Hugging Face Datasets y devuelve un DataFrame de pandas. Opcionalmente limita filas (`max_rows`) y guarda una copia en csv/json/parquet/xlsx (`filename`). |
| `src/dataset_summarizer.py` | Clase que recibe un DataFrame y genera un resumen en `.txt` (dimensiones, tipos, nulos, duplicados, estadísticas y frecuencias). |
| `notebooks/EA1_carga_dataset.ipynb` | Notebook que coordina las dos clases y aplica 6 transformaciones. |
| `datos/iris_original.csv` | Copia del dataset original guardada por `HFDatasetLoader`. |
| `output/resumen_iris.txt` | Resumen generado a partir del DataFrame transformado. |
| `requirements.txt` | Dependencias del proyecto. |

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter notebook notebooks/EA1_carga_dataset.ipynb
```

Ejecutar todas las celdas en orden (Kernel → Restart & Run All).

## Ejemplo de uso de las clases

```python
from src.hf_dataset_loader import HFDatasetLoader
from src.dataset_summarizer import DatasetSummarizer

loader = HFDatasetLoader("scikit-learn/iris", split="train", filename="datos/iris.csv")
df = loader.load()

DatasetSummarizer(df, nombre="iris").save("output/resumen.txt")
```

Los datasets públicos no necesitan token. Para uno privado, inicia sesión con `hf auth login` o pasa `token=...` (nunca subas el token a GitHub).

Dataset usado: [`scikit-learn/iris`](https://huggingface.co/datasets/scikit-learn/iris).
