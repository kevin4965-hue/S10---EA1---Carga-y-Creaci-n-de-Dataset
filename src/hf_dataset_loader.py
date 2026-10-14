"""Carga de datasets desde Hugging Face Datasets como DataFrame de pandas."""

from pathlib import Path

from datasets import load_dataset, get_dataset_split_names


class HFDatasetLoader:
    """Obtiene un dataset de Hugging Face y lo devuelve como DataFrame.

    Ejemplo:
        loader = HFDatasetLoader("scikit-learn/iris", split="train",
                                 filename="datos/iris.csv")
        df = loader.load()
    """

    FORMATOS = ("csv", "json", "parquet", "xlsx")

    def __init__(self, dataset_id, split="train", config_name=None, token=None,
                 cache_dir=None, max_rows=None, filename=None, file_type=None):
        """
        Args:
            dataset_id: identificador del dataset en Hugging Face (ej. "scikit-learn/iris").
            split: partición a cargar ("train", "test", "validation", ...).
            config_name: configuración/subconjunto del dataset, si la tiene.
            token: token de Hugging Face, solo para datasets privados o restringidos.
            cache_dir: carpeta de caché de Hugging Face (opcional).
            max_rows: cantidad máxima de filas a cargar (opcional).
            filename: ruta donde guardar una copia del DataFrame (opcional).
            file_type: formato de la copia: csv, json, parquet o xlsx. Si no se
                indica, se toma de la extensión de filename.
        """
        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows debe ser mayor o igual que cero.")
        if file_type is not None and filename is None:
            raise ValueError("filename es obligatorio cuando se indica file_type.")

        self.dataset_id = dataset_id
        self.split = split
        self.config_name = config_name
        self.token = token
        self.cache_dir = cache_dir
        self.max_rows = max_rows
        self.filename = Path(filename) if filename is not None else None
        self.file_type = self._normalize_file_type(file_type)
        self.dataset = None
        self.dataframe = None

    def available_splits(self):
        """Devuelve la lista de splits disponibles para el dataset."""
        return get_dataset_split_names(self.dataset_id, self.config_name, token=self.token)

    def _connect(self):
        """Se conecta a Hugging Face y descarga el split indicado."""
        try:
            self.dataset = load_dataset(
                self.dataset_id,
                self.config_name,
                split=self.split,
                token=self.token,
                cache_dir=self.cache_dir,
            )
        except ValueError as error:
            raise ValueError(
                f"No se pudo cargar el split '{self.split}' de '{self.dataset_id}'. "
                f"Splits disponibles: {self.available_splits()}"
            ) from error
        if self.max_rows is not None:
            self.dataset = self.dataset.select(range(min(self.max_rows, len(self.dataset))))
        return self.dataset

    def load(self):
        """Carga el dataset, lo guarda si se indicó filename y devuelve el DataFrame."""
        if self.dataset is None:
            self._connect()
        self.dataframe = self.dataset.to_pandas()
        if self.filename is not None:
            self.save(self.filename, self.file_type)
        return self.dataframe

    def save(self, filename, file_type=None):
        """Guarda el DataFrame cargado en csv, json, parquet o xlsx y devuelve la ruta."""
        if self.dataframe is None:
            raise RuntimeError("Llama a load() antes de guardar.")
        ruta = Path(filename)
        formato = self._normalize_file_type(file_type) or self._normalize_file_type(ruta.suffix)
        if formato is None:
            raise ValueError(f"Indica file_type o usa una extensión válida: {', '.join(self.FORMATOS)}.")

        ruta.parent.mkdir(parents=True, exist_ok=True)
        if formato == "csv":
            self.dataframe.to_csv(ruta, index=False)
        elif formato == "json":
            self.dataframe.to_json(ruta, orient="records", force_ascii=False)
        elif formato == "parquet":
            self.dataframe.to_parquet(ruta, index=False)
        elif formato == "xlsx":
            self.dataframe.to_excel(ruta, index=False)
        return ruta

    def info(self):
        """Devuelve un diccionario con información básica de la carga."""
        if self.dataframe is None:
            self.load()
        return {
            "dataset_id": self.dataset_id,
            "config": self.config_name,
            "split": self.split,
            "filas": self.dataframe.shape[0],
            "columnas": self.dataframe.shape[1],
            "guardado_en": str(self.filename) if self.filename else None,
        }

    @classmethod
    def _normalize_file_type(cls, file_type):
        """Convierte '.CSV', 'excel', etc. al nombre de formato soportado."""
        if not file_type:
            return None
        formato = file_type.lower().lstrip(".")
        if formato == "excel":
            formato = "xlsx"
        if formato not in cls.FORMATOS:
            raise ValueError(f"Formato no soportado: {file_type}. Usa: {', '.join(cls.FORMATOS)}.")
        return formato
