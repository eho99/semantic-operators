import numpy as np
from typing import Tuple
import litellm
from litellm import embedding
from litellm.caching.caching import Cache
from src.config import AZURE_API_KEY, AZURE_API_BASE, EMBEDDING_DEPLOYMENT, AZURE_API_VERSION 

class Embedder:
    def __init__(self, client=None, model_name: str = EMBEDDING_DEPLOYMENT, use_cache=True):
        if use_cache:
            litellm.cache = Cache(type="disk")

        if client is None:
            # Initialize default configuration for Azure
            self.client = {
                'model': model_name,
                'api_key': AZURE_API_KEY,
                'api_base': AZURE_API_BASE,
                'api_version': AZURE_API_VERSION
            }
        else:
            self.client = client
        self.model_name = model_name
        self._embedding_cost = 0.0
        self._total_tokens = 0

    def embed_documents(self, df, column_name: str, batch_size: int = 100) -> Tuple[np.ndarray, float]:
        """
        Generates embeddings for texts in a specified column of a DataFrame.
        Processes the data in batches and returns embeddings and total cost.
        """
        embeddings = []
        total_cost = 0.0
        for i in range(0, len(df), batch_size):
            batch = df[column_name].iloc[i:i + batch_size].tolist()
            batch_embeddings = []
            for text in batch:
                embedding_vector, cost = self.get_embedding(text)
                batch_embeddings.append(embedding_vector)
                total_cost += cost
            embeddings.extend(batch_embeddings)
        return np.array(embeddings), total_cost
        
    def get_embedding(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Retrieves the embedding for a given text using the Azure model.
        Tracks token usage and cost per call.
        """
        response = embedding(
            model=self.client['model'],
            input=[text],
            api_key=self.client['api_key'],
            api_base=self.client['api_base'],
            api_version=self.client['api_version'],
            caching=True,
        )
        
        # Track tokens and cost based on the response metadata.
        token_count = response.usage.total_tokens
        self._total_tokens += token_count
        # Adjust the rate to match your Azure pricing.
        embedding_cost = 0.00002 * (token_count / 1000) # $0.00002 per 1000 tokens
        self._embedding_cost += embedding_cost
        embedded_text = response.data[0]['embedding']

        return embedded_text, embedding_cost

    def get_total_cost(self) -> float:
        """
        Returns the cumulative cost for embedding API calls.
        """
        return self._embedding_cost

    def get_total_tokens(self) -> int:
        """
        Returns the cumulative token count for embedding API calls.
        """
        return self._total_tokens
