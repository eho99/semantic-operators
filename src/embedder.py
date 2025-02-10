import numpy as np
# from litellm import OpenAIClient  # or your specific litellm import
from litellm import embedding

from src.config import AZURE_API_KEY, AZURE_API_BASE, EMBEDDING_DEPLOYMENT, AZURE_API_VERSION 

class Embedder:
    def __init__(self, client=None, model_name: str = EMBEDDING_DEPLOYMENT):
        self.client = client
        self.model_name = model_name
        self._embedding_cost = 0.0
        self._total_tokens = 0

    def embed_documents(self, df, column_name: str, batch_size: int = 100) -> np.ndarray:
        """
        Generates embeddings for texts in a specified column of a dataframe.
        Args:
            df: pandas DataFrame containing the texts
            column_name: name of the column containing texts to embed
            batch_size: number of texts to process in each batch
        Returns:
            numpy array of embeddings
        """
        embeddings = []
        for i in range(0, len(df), batch_size):
            batch = df[column_name].iloc[i:i + batch_size].tolist()
            batch_embeddings = [self.get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Retrieves the embedding for a given text using the Azure model.
        """
        embedded_text = None
        if self.client:
            pass
        else:
            response = embedding(
                    model=self.model_name, 
                    input=[text], 
                    api_key=AZURE_API_KEY, 
                    api_base=AZURE_API_BASE, 
                    api_version=AZURE_API_VERSION
                )
                
            # Track tokens and cost
            token_count = response.usage.total_tokens
            self._total_tokens += token_count
            # Azure pricing for embeddings (adjust as needed)
            self._embedding_cost += (token_count / 1000) * 0.0001  # FIXME: get cost from the litellm request
            embedded_text = response.data[0]['embedding']

        return embedded_text

    def get_total_cost(self) -> float:
        return self._embedding_cost

    def get_total_tokens(self) -> int:
        return self._total_tokens            
