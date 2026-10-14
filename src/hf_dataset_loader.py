"""Carga de datasets desde Hugging Face Datasets como DataFrame de pandas."""

from datasets import load_dataset, get_dataset_split_names


class HFDatasetLoader:
    """Obtiene un dataset de Hugging Face y lo devuelve como DataFrame.

    Ejemplo:
        loader = HFDatasetLoader("scikit-learn/iris", split="train")
        df = loader.load()
    """

    def __init__(self, dataset_id, split="train", config_name=None):
        """
        Args:
            dataset_id: identificador del dataset en Hugging Face (ej. "scikit-learn/iris").
            split: partición a cargar ("train", "test", "validation", ...).
            config_name: configuración/subconjunto del dataset, si la tiene.
        """
        self.dataset_id = dataset_id
        self.split = split
        self.config_name = config_name
        self.dataset = None
        self.dataframe = None

    def available_splits(self):
        """Devuelve la lista de splits disponibles para el dataset."""
        return get_dataset_split_names(self.dataset_id, self.config_name)

    def _connect(self):
        """Se conecta a Hugging Face y descarga el split indicado."""
        try:
            self.dataset = load_dataset(
                self.dataset_id, self.config_name, split=self.split
            )
        except ValueError as error:
            raise ValueError(
                f"No se pudo cargar el split '{self.split}' de '{self.dataset_id}'. "
                f"Splits disponibles: {self.available_splits()}"
            ) from error
        return self.dataset

    def load(self):
        """Carga el dataset y lo devuelve como un DataFrame de pandas."""
        if self.dataset is None:
            self._connect()
        self.dataframe = self.dataset.to_pandas()
        return self.dataframe

    def info(self):
        """Devuelve un diccionario con información básica de la carga."""
        if self.dataframe is None:
            self.load()
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "filas": self.dataframe.shape[0],
            "columnas": self.dataframe.shape[1],
        }
