import time
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import lotus
import litellm
from lotus.models import LM
from lotus.sem_ops.sem_join import sem_join
from dotenv import load_dotenv

from src.utils import calculate_cosine_similarity
from src.embedder import Embedder

load_dotenv()

class SemanticJoin:
    def __init__(self, data_loader, embedder_client=None, model_name=None):
        self.data_loader = data_loader
        self.embedder = Embedder(embedder_client)
        
        model = model_name or os.getenv("LLM_DEPLOYMENT", "gpt-4o-mini")
        self.lm = LM(model=model)
        lotus.settings.configure(lm=self.lm)

    def compute_similarity_matrix(self, descriptions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        descriptions_norm = descriptions / np.linalg.norm(descriptions, axis=1)[:, np.newaxis]
        labels_norm = labels / np.linalg.norm(labels, axis=1)[:, np.newaxis]
        similarity_matrix = np.dot(descriptions_norm, labels_norm.T)
        return similarity_matrix

    def perform_embedding_join(self, df_desc, df_labels, id_col: str, desc_col: str, label_col: str) -> Tuple[pd.DataFrame, float]:
        """
        Perform embedding-based join and include sample id.
        
        Returns:
            A pandas DataFrame with columns: sample_id, desc_text, matched_label, similarity_score,
            and the combined cost.
        """
        print("Embedding documents")
        desc_embeddings, desc_cost = self.embedder.embed_documents(df_desc, desc_col)
        label_embeddings, label_cost = self.embedder.embed_documents(df_labels, label_col)

        print("Computing similarity matrix")
        sim_matrix = self.compute_similarity_matrix(desc_embeddings, label_embeddings)
        
        sample_ids = df_desc[id_col].tolist()
        descriptions = df_desc[desc_col].tolist()
        labels = df_labels[label_col].tolist()
        best_matches = np.argmax(sim_matrix, axis=1)
        best_similarities = np.max(sim_matrix, axis=1)
        
        # Create a list of dictionaries for each join result
        join_records = []
        for sample_id, desc, best_idx, similarity in zip(sample_ids, descriptions, best_matches, best_similarities):
            join_records.append({
                "sample_id": sample_id,
                "desc_text": desc,
                "matched_label": labels[best_idx],
                "similarity_score": float(similarity)
            })
        
        results_df = pd.DataFrame(join_records)
        print("Join Results", results_df)
        return results_df, desc_cost + label_cost

    def perform_llm_join(self, data_a: pd.DataFrame, data_b: pd.DataFrame, id_col: str, col_a: str, col_b: str, prompt_template: Optional[str] = None) -> Tuple[pd.DataFrame, float]:
        """
        Perform LLM-powered join and include sample id.
        
        Returns:
            A pandas DataFrame with columns: sample_id, value_a, value_b, confidence_score,
            and the cost for the LLM join.
        """
        try:
            print("Performing LLM join")
            initial_cost = self.lm.stats.total_usage.total_cost

            df_a = data_a.rename(columns={col_a: "value_a"})
            df_b = data_b.rename(columns={col_b: "value_b"})

            if not prompt_template:
                prompt_template = "Are {value_a} and {value_b} semantically related?"

            res = sem_join(
                l1=df_a['value_a'],
                l2=df_b['value_b'],
                ids1=list(range(len(df_a))),
                ids2=list(range(len(df_b))),
                col1_label='value_a',
                col2_label='value_b',
                model=self.lm,
                user_instruction=prompt_template,
                default=True,
            )
            print(res)
            
            sample_ids = data_a[id_col].tolist()
            join_records = []
            for idx_a, idx_b, _ in res.join_results:
                join_records.append({
                    "sample_id": sample_ids[idx_a],
                    "matched_desc": df_a['value_a'].iloc[idx_a],
                    "matched_label": df_b['value_b'].iloc[idx_b],
                    # "confidence_score": float(confidence)
                })

            join_cost = self.lm.stats.total_usage.total_cost - initial_cost
            results_df = pd.DataFrame(join_records)
            print("LLM join cost:", join_cost)
            print("LLM join results:", results_df)
            return results_df, join_cost
            
        except Exception as e:
            raise Exception(f"LLM join failed: {str(e)}")

    def benchmark(self, join_method, data_a, data_b, truth, **kwargs) -> dict:
        """
        Benchmark the performance of a join method.
        
        Args:
            join_method: 'embedding' or 'llm'
            data_a: First dataframe
            data_b: Second dataframe
            truth: Ground truth DataFrame. It could map left id to right id, left desc to right id,
                   left id to right label, or left desc to right label.
            **kwargs: Additional arguments for join methods
        
        Returns:
            Dictionary containing execution_time, cost, accuracy, total_pairs, and matched_pairs.
        """
        
        metrics = {
            'execution_time': 0,
            'cost': 0,
            'accuracy': 0,
            'total_pairs': 0,
            'matched_pairs': 0
        }
        start_time = time.time()
        
        try:
            if join_method == 'embedding':
                results_df, cost = self.perform_embedding_join(data_a, data_b, **kwargs)
                metrics["cost"] = cost
            elif join_method == 'llm':
                results_df, cost = self.perform_llm_join(data_a, data_b, **kwargs)
                metrics["cost"] = cost
            else:
                raise ValueError(f"Unknown join method: {join_method}")
            
            print("Truth data:", truth)
            metrics['execution_time'] = time.time() - start_time
            metrics['total_pairs'] = len(results_df)
            
            # Determine common columns between the join results and truth dataframe for matching.
            truth_cols = truth.columns.tolist()
            common_cols = [col for col in truth_cols if col in results_df.columns]
            if common_cols:
                # Create combined keys for both results_df and truth for accurate matching
                # TODO: more results returned are just dropped right now
                results_df['combined_key'] = results_df[common_cols].apply(lambda row: '_'.join(row.astype(str)), axis=1)
                truth['combined_key'] = truth[common_cols].apply(lambda row: '_'.join(row.astype(str)), axis=1)
                
                common_keys = set(results_df['combined_key']).intersection(truth['combined_key'])
                
                # Calculate matched_count directly from common_keys
                matched_count = len(common_keys)
            else:
                matched_count = 0
            metrics['matched_pairs'] = matched_count
            
            if len(truth) > 0:
                metrics['accuracy'] = matched_count / len(truth)
            else:
                metrics['accuracy'] = 0

            print("Benchmark join results:", results_df)
            
        except Exception as e:
            print(f"Benchmark failed: {str(e)}")
            
        return metrics