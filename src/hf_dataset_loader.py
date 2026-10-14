"""Carga de datasets desde Hugging Face Datasets como DataFrame de pandas."""

from pathlib import Path

from datasets import load_dataset, get_dataset_split_names


class HFDatasetLoader:
    """Obtiene un dataset de Hugging Face y lo devuelve como DataFrame.

    Flujo general:
        1. _connect() descarga el split pedido (y lo recorta si hay max_rows).
        2. load() lo convierte a DataFrame y, si se indicó filename, lo guarda.
        3. save() escribe el DataFrame en csv, json, parquet o xlsx.

    Ejemplo:
        loader = HFDatasetLoader("scikit-learn/iris", split="train",
                                 filename="datos/iris.csv")
        df = loader.load()
    """

    # Formatos de archivo que save() sabe escribir.
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

        Raises:
            ValueError: si max_rows es negativo, si se pasa file_type sin
                filename o si file_type no es un formato soportado.
        """
        # Validaciones tempranas para fallar antes de descargar nada.
        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows debe ser mayor o igual que cero.")
        if file_type is not None and filename is None:
            raise ValueError("filename es obligatorio cuando se indica file_type.")

        # Parámetros de la carga.
        self.dataset_id = dataset_id
        self.split = split
        self.config_name = config_name
        self.token = token
        self.cache_dir = cache_dir
        self.max_rows = max_rows

        # Parámetros del guardado opcional.
        self.filename = Path(filename) if filename is not None else None
        self.file_type = self._normalize_file_type(file_type)

        # Resultados: se llenan al llamar a _connect() y load().
        self.dataset = None     # objeto datasets.Dataset de Hugging Face
        self.dataframe = None   # el mismo contenido como pandas.DataFrame

    def available_splits(self):
        """Devuelve la lista de splits disponibles para el dataset."""
        return get_dataset_split_names(self.dataset_id, self.config_name, token=self.token)

    def _connect(self):
        """Se conecta a Hugging Face y descarga el split indicado.

        Returns:
            El datasets.Dataset descargado (recortado a max_rows si aplica).
        """
        try:
            self.dataset = load_dataset(
                self.dataset_id,
                self.config_name,
                split=self.split,
                token=self.token,
                cache_dir=self.cache_dir,
            )
        except ValueError as error:
            # El error más común es pedir un split que no existe: se relanza
            # con un mensaje que muestra los splits válidos para ese dataset.
            raise ValueError(
                f"No se pudo cargar el split '{self.split}' de '{self.dataset_id}'. "
                f"Splits disponibles: {self.available_splits()}"
            ) from error

        # Recorta a las primeras max_rows filas; min() evita salirse del rango
        # cuando el dataset tiene menos filas que las pedidas.
        if self.max_rows is not None:
            self.dataset = self.dataset.select(range(min(self.max_rows, len(self.dataset))))
        return self.dataset

    def load(self):
        """Carga el dataset, lo guarda si se indicó filename y devuelve el DataFrame.

        Returns:
            pandas.DataFrame con los datos del split.
        """
        # Solo descarga la primera vez; las siguientes reutiliza self.dataset.
        if self.dataset is None:
            self._connect()
        self.dataframe = self.dataset.to_pandas()

        # Si se configuró un archivo de salida en el constructor, se guarda la copia.
        if self.filename is not None:
            self.save(self.filename, self.file_type)
        return self.dataframe

    def save(self, filename, file_type=None):
        """Guarda el DataFrame cargado en csv, json, parquet o xlsx y devuelve la ruta.

        Args:
            filename: ruta del archivo de salida.
            file_type: formato a usar; si es None se deduce de la extensión.

        Returns:
            Path del archivo escrito.

        Raises:
            RuntimeError: si todavía no se ha llamado a load().
            ValueError: si no se puede determinar un formato válido.
        """
        if self.dataframe is None:
            raise RuntimeError("Llama a load() antes de guardar.")
        ruta = Path(filename)

        # Prioridad: file_type explícito; si no hay, la extensión del archivo.
        formato = self._normalize_file_type(file_type) or self._normalize_file_type(ruta.suffix)
        if formato is None:
            raise ValueError(f"Indica file_type o usa una extensión válida: {', '.join(self.FORMATOS)}.")

        # Crea la carpeta destino si no existe.
        ruta.parent.mkdir(parents=True, exist_ok=True)

        # index=False para no escribir el índice de pandas como columna extra.
        if formato == "csv":
            self.dataframe.to_csv(ruta, index=False)
        elif formato == "json":
            # orient="records" -> lista de objetos {columna: valor};
            # force_ascii=False conserva tildes y ñ.
            self.dataframe.to_json(ruta, orient="records", force_ascii=False)
        elif formato == "parquet":
            self.dataframe.to_parquet(ruta, index=False)
        elif formato == "xlsx":
            self.dataframe.to_excel(ruta, index=False)
        return ruta

    def info(self):
        """Devuelve un diccionario con información básica de la carga.

        Si el dataset aún no se ha cargado, llama a load() primero.
        """
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
        """Convierte '.CSV', 'excel', etc. al nombre de formato soportado.

        Returns:
            El formato normalizado (ej. "csv") o None si file_type está vacío.

        Raises:
            ValueError: si el formato no está en FORMATOS.
        """
        # None o "" (p. ej. un archivo sin extensión) -> sin formato.
        if not file_type:
            return None
        # Minúsculas y sin punto inicial: ".CSV" -> "csv".
        formato = file_type.lower().lstrip(".")
        # Alias aceptado para Excel.
        if formato == "excel":
            formato = "xlsx"
        if formato not in cls.FORMATOS:
            raise ValueError(f"Formato no soportado: {file_type}. Usa: {', '.join(cls.FORMATOS)}.")
        return formato
