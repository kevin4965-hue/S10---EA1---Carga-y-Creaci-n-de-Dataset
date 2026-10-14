"""Generación de un resumen en texto (.txt) a partir de un DataFrame de pandas."""

from datetime import datetime
from pathlib import Path

import pandas as pd


class DatasetSummarizer:
    """Recibe un DataFrame y genera un resumen legible en un archivo .txt.

    Ejemplo:
        summarizer = DatasetSummarizer(df, nombre="iris transformado")
        summarizer.save("output/resumen.txt")
    """

    def __init__(self, dataframe, nombre="dataset"):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("DatasetSummarizer espera un pandas.DataFrame")
        self.df = dataframe
        self.nombre = nombre

    def dimensions(self):
        """Cantidad de filas y columnas."""
        filas, columnas = self.df.shape
        return f"Filas: {filas}\nColumnas: {columnas}"

    def columns_info(self):
        """Nombre, tipo, nulos y valores únicos de cada columna."""
        tabla = pd.DataFrame({
            "tipo": self.df.dtypes.astype(str),
            "nulos": self.df.isnull().sum(),
            "% nulos": (self.df.isnull().mean() * 100).round(2),
            "valores_unicos": self.df.nunique(),
        })
        return tabla.rename_axis("columna").to_string()

    def nulls_info(self):
        """Indica si hay valores nulos y en qué columnas."""
        total = int(self.df.isnull().sum().sum())
        if total == 0:
            return "No hay valores nulos en el DataFrame."
        columnas = self.df.columns[self.df.isnull().any()].tolist()
        return f"Total de valores nulos: {total}\nColumnas con nulos: {', '.join(columnas)}"

    def numeric_stats(self):
        """Estadísticas descriptivas de las columnas numéricas."""
        numericas = self.df.select_dtypes(include="number")
        if numericas.empty:
            return "No hay columnas numéricas."
        return numericas.describe().round(3).to_string()

    def categorical_stats(self):
        """Frecuencias de las columnas categóricas (texto/categoría)."""
        categoricas = self.df.select_dtypes(include=["object", "category"])
        if categoricas.empty:
            return "No hay columnas categóricas."
        bloques = []
        for columna in categoricas.columns:
            conteo = self.df[columna].value_counts().head(10).to_string()
            bloques.append(f"[{columna}]\n{conteo}")
        return "\n\n".join(bloques)

    def build_summary(self):
        """Arma el texto completo del resumen."""
        secciones = [
            ("1. DIMENSIONES", self.dimensions()),
            ("2. COLUMNAS: TIPOS, NULOS Y VALORES ÚNICOS", self.columns_info()),
            ("3. VALORES NULOS", self.nulls_info()),
            ("4. ESTADÍSTICAS DESCRIPTIVAS (NUMÉRICAS)", self.numeric_stats()),
            ("5. FRECUENCIAS (CATEGÓRICAS)", self.categorical_stats()),
            ("6. PRIMERAS 5 FILAS", self.df.head().to_string()),
        ]
        encabezado = (
            f"RESUMEN DEL DATASET: {self.nombre}\n"
            f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            + "=" * 70
        )
        cuerpo = "\n\n".join(f"{titulo}\n{'-' * 70}\n{texto}" for titulo, texto in secciones)
        return f"{encabezado}\n\n{cuerpo}\n"

    def save(self, ruta="resumen_dataset.txt"):
        """Guarda el resumen en un archivo .txt y devuelve la ruta."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(self.build_summary(), encoding="utf-8")
        return ruta
