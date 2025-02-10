import litellm
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import time
import os
import lotus
from lotus.models import LM
from dotenv import load_dotenv

from src.utils import calculate_cosine_similarity
from src.embedder import Embedder

load_dotenv()

class SemanticJoin:
    def __init__(self, data_loader, embedder_client=None):
        self.data_loader = data_loader
        self.embedder = Embedder(embedder_client)

    def compute_similarity_matrix(self, descriptions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between descriptions and labels using their embeddings.
        Args:
            descriptions: numpy array of description embeddings (n_desc x embedding_dim)
            labels: numpy array of label embeddings (n_labels x embedding_dim)
        Returns:
            similarity_matrix: numpy array of shape (n_desc x n_labels)
        """
        # Compute dot product between normalized vectors (cosine similarity)
        descriptions_norm = descriptions / np.linalg.norm(descriptions, axis=1)[:, np.newaxis]
        labels_norm = labels / np.linalg.norm(labels, axis=1)[:, np.newaxis]
        similarity_matrix = np.dot(descriptions_norm, labels_norm.T)
        return similarity_matrix

    def perform_embedding_join(self, df_desc, df_labels, desc_col: str, label_col: str) -> List[Tuple[str, str, float]]:
        """
        Perform embedding-based join between descriptions and labels using dataframes.
        Args:
            df_desc: DataFrame containing descriptions
            df_labels: DataFrame containing labels
            desc_col: Column name for descriptions
            label_col: Column name for labels
        Returns:
            List of (description, matched_label, similarity_score) tuples
        """
        print("embedding documents")
        # Get embeddings for descriptions and labels using embedder
        desc_embeddings = self.embedder.embed_documents(df_desc, desc_col)
        label_embeddings = self.embedder.embed_documents(df_labels, label_col)

        # print(desc_embeddings)
        # print(label_embeddings)
        
        # Compute similarity matrix
        print("computing similarity matrix")
        sim_matrix = self.compute_similarity_matrix(desc_embeddings, label_embeddings)
        
        # Find best matches
        join_results = []
        descriptions = df_desc[desc_col].tolist()
        labels = df_labels[label_col].tolist()

        # print(sim_matrix)
        
        best_matches = np.argmax(sim_matrix, axis=1)
        best_similarities = np.max(sim_matrix, axis=1)
        
        for i, (desc, best_idx, similarity) in enumerate(zip(descriptions, best_matches, best_similarities)):
            join_results.append((desc, labels[best_idx], float(similarity)))
        
        print(join_results)
        return join_results

    def perform_llm_join(self, data_a: pd.DataFrame, data_b: pd.DataFrame, col_a: str, col_b: str, prompt_template: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """
        Perform LLM-powered join using Lotus framework.
        
        Args:
            data_a: First DataFrame
            data_b: Second DataFrame
            col_a: Column name from first DataFrame
            col_b: Column name from second DataFrame
            prompt_template: Optional custom prompt template. If None, uses default:
                        "Are {col_a} and {col_b} semantically related?"
        
        Returns:
            List of tuples (value_a, value_b, confidence_score)
        """
        try:
            # Configure Lotus LM if not already set
            if not hasattr(self, 'lm'):
                self.lm = LM(model=os.getenv("LLM_DEPLOYMENT", "gpt-4o-mini"))
                lotus.settings.configure(lm=self.lm)
            
            # Rename columns to match input DataFrames
            df_a = data_a.rename(columns={col_a: "value_a"})
            df_b = data_b.rename(columns={col_b: "value_b"})
            
            # Default prompt template if none provided
            if not prompt_template:
                prompt_template = "Are {value_a} and {value_b} semantically related?"
            
            # Perform semantic join using Lotus
            results = df_a.sem_join(
                df_b,
                prompt_template,
            )

            # Store results for later processing
            print(results)
            # print(stats)
            self.lm.print_total_usage()
            
            # Convert results to required format
            join_results = []
            for _, row in results.iterrows():
                join_results.append((
                    row['value_a'],
                    row['value_b'],
                    # float(row['confidence'])  # Lotus returns confidence scores
                ))
            
            print(join_results)
            return join_results
            
        except Exception as e:
            raise Exception(f"LLM join failed: {str(e)}")

    def benchmark(self, join_method, data_a, data_b, truth, **kwargs) -> dict:
        """
        Benchmark the performance of a join method.
        Args:
            join_method: Method to benchmark ('embedding' or 'llm')
            data_a: First dataframe
            data_b: Second dataframe
            **kwargs: Additional arguments for join methods
        Returns:
            Dictionary containing time, cost, and accuracy metrics
        """
        
        metrics = {
            'execution_time': 0,
            # 'cost': 0,
            'accuracy': 0,
            'total_pairs': 0,
            'matched_pairs': 0
        }
        
        start_time = time.time()
        
        try:
            if join_method == 'embedding':
                # Track embedding costs through embedder
                # initial_cost = self.embedder.get_total_cost()
                results = self.perform_embedding_join(data_a, data_b, **kwargs)
                # metrics['cost'] = self.embedder.get_total_cost() - initial_cost
                
            elif join_method == 'llm':
                # Track LLM costs
                # initial_cost = litellm.total_cost
                results = self.perform_llm_join(data_a, data_b, **kwargs)
                # metrics['cost'] = litellm.total_cost - initial_cost
                # metrics['cost'] = 0  # FIXME: LLM cost tracking not implemented yet
                
            else:
                raise ValueError(f"Unknown join method: {join_method}")
            
            metrics['execution_time'] = time.time() - start_time
            metrics['total_pairs'] = len(results)
            metrics['matched_pairs'] = len([(a, b) for a, b in results if (a, b) in truth])  # Count matching pairs with truth

            print("Benchmark results:", results)
            
            # Calculate accuracy if ground truth is provided
            if truth is not None:
                # Convert truth to set of pairs if not already
                truth_pairs = set(truth) if isinstance(truth, set) else set(truth)
                # Convert results to comparable format (removing confidence scores)
                result_pairs = set((a, b) for a, b in results)
                # Calculate accuracy as intersection of correct pairs
                correct_matches = result_pairs & truth_pairs
                metrics['accuracy'] = len(correct_matches) / len(truth_pairs)
                
        except Exception as e:
            print(f"Benchmark failed: {str(e)}")
            
        return metrics