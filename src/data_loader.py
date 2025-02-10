from src.utils import normalize_text
import pandas as pd

class DataLoader:
    def __init__(self, sample_filename: str, truth_filename: str, labels_filename: str):
        self.sample_filename = sample_filename
        self.truth_filename = truth_filename
        self.labels_filename = labels_filename

    def load_data(self):
        sample_data = self._load_from_json(self.sample_filename)
        truth_data = self._load_from_json(self.truth_filename)
        labels_data = self._load_from_json(self.labels_filename)

        return sample_data, truth_data, labels_data

    def preprocess_data(self, data, text_columns=None):
        """
        Preprocess data by normalizing specified text columns.
        Args:
            data: List of dictionaries or pandas DataFrame
            text_columns: List of column names to normalize. If None, normalizes all string columns.
        Returns:
            pandas DataFrame with normalized text
        """
        
        # Convert to DataFrame if not already
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        
        # If no columns specified, find all object (string) columns
        if text_columns is None:
            text_columns = df.select_dtypes(include=['object']).columns
        
        # Normalize specified text columns
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: normalize_text(x) if isinstance(x, str) else x)
        
        return df

    def _load_from_json(self, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data