# Proyecto Integrado 3 — EA1: Carga y creación de dataset

Flujo de la actividad:

```
HFDatasetLoader (src/hf_dataset_loader.py) → DataFrame → transformaciones (notebook)
                                                      → DatasetSummarizer (src/dataset_summarizer.py) → output/resumen_iris.txt
```

## Estructura

| Archivo | Descripción |
|---|---|
| `src/hf_dataset_loader.py` | Clase que se conecta a Hugging Face Datasets y devuelve un DataFrame de pandas. |
| `src/dataset_summarizer.py` | Clase que recibe un DataFrame y genera un resumen en `.txt`. |
| `notebooks/EA1_carga_dataset.ipynb` | Notebook que coordina las dos clases y aplica las transformaciones. |
| `output/resumen_iris.txt` | Resumen generado a partir del DataFrame transformado. |
| `requirements.txt` | Dependencias del proyecto. |

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter notebook notebooks/EA1_carga_dataset.ipynb
```

Ejecutar todas las celdas en orden (Kernel → Restart & Run All).

Dataset usado: [`scikit-learn/iris`](https://huggingface.co/datasets/scikit-learn/iris).
