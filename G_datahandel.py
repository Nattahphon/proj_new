import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class DataHandler:
    _instance = None
    _data = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataHandler, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def load_data(self) -> pd.DataFrame:
        """Load and standardize data from a file."""
        dataset_path = os.getenv("DATASET_PATH")
        if not dataset_path or not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}.")

        _, ext = os.path.splitext(dataset_path)
        if ext == ".csv":
            self._data = pd.read_csv(dataset_path)
        elif ext in [".xls", ".xlsx"]:
            self._data = pd.read_excel(dataset_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Standardize column names
        self._data.columns = self._data.columns.str.lower().str.strip().str.replace(" ", "_")
        print(f"Data loaded. Columns: {', '.join(self._data.columns)}")
        return self._data

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess numeric-like columns."""
        if self._data is None:
            raise ValueError("Data not loaded.")

        # Identify numeric-like columns
        numeric_like_cols = [
            col for col in self._data.columns 
            if self._data[col].dtype == 'object' and self._data[col].str.contains(r"[\d,.$€¥-]").any()
        ]

        for col in numeric_like_cols:
            try:
                self._data[col] = self._data[col].str.replace(r"[^\d.-]", "", regex=True)
                self._data[col] = pd.to_numeric(self._data[col], errors="coerce")
            except Exception as e:
                print(f"Error processing column {col}: {e}")

        print("Preprocessing complete.")
        return self._data

    def get_data(self) -> pd.DataFrame:
        """Retrieve the loaded data."""
        if self._data is None:
            raise ValueError("Data not loaded.")
        return self._data
