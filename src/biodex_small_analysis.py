from semantic_join import SemanticJoin
from data_loader import DataLoader
import pandas as pd
import logging
import os
from datetime import datetime

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"biodex_analysis_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger(__name__)

def run_biodex_analysis():
    logger = setup_logging()
    logger.info("Starting Biodex sample analysis...")

    # Initialize DataLoader with biodex data files
    logger.info("Loading data from biodex_small...")
    data_loader = DataLoader(
        sample_filename="data/biodex_small/sample.json",
        truth_filename="data/biodex_small/truth.json",
        labels_filename="data/biodex_small/labels.json"
    )

    # Load and preprocess the data
    sample_data, truth_data, labels_data = data_loader.load_data()

    # Convert to DataFrames
    logger.info("Processing sample descriptions...")
    sample_df = pd.DataFrame(sample_data)
    
    logger.info("Processing labels...")
    labels_df = pd.DataFrame(labels_data)
    
    # Extract ground truth pairs from truth data and convert to DataFrame.
    # Ground truth should map sample_id to the expected reaction label.
    logger.info("Processing ground truth data...")
    ground_truth = []
    for item in truth_data:
        if 'id' in item and 'ground_truth_reactions' in item:
            sample_id = item['id']
            for reaction in item['ground_truth_reactions']:
                ground_truth.append((sample_id, reaction))
    
    logger.info(f"Found {len(ground_truth)} ground truth pairs")
    # Convert ground truth to a DataFrame with columns matching the join results.
    # For both embedding and LLM join the common columns will be "sample_id" and "matched_label".
    ground_truth_df = pd.DataFrame(ground_truth, columns=["sample_id", "matched_label"])

    # Initialize semantic join
    logger.info("Initializing SemanticJoin...")
    join_op = SemanticJoin(data_loader)
    
    logger.debug(f"Sample DataFrame:\n{sample_df}")
    
    # Benchmark embedding join and pass the sample id column for tracking
    logger.info("Benchmarking embedding join...")
    embedding_metrics = join_op.benchmark(
        join_method='embedding',
        data_a=sample_df,
        data_b=labels_df,
        truth=ground_truth_df,
        id_col='id',
        desc_col='fulltext_processed',
        label_col='reaction'
    )
    embedding_cost = embedding_metrics['cost']
    logger.info(f"Embedding Join Metrics: {embedding_metrics}")

    # Benchmark LLM join and pass the sample id column for tracking
    logger.info("Benchmarking LLM join...")
    llm_metrics = join_op.benchmark(
        join_method='llm',
        data_a=sample_df,
        data_b=labels_df,
        truth=ground_truth_df,
        id_col='id',
        col_a='fulltext_processed',
        col_b='reaction'
    )
    llm_cost = llm_metrics['cost']
    logger.info(f"LLM Join Metrics: {llm_metrics}")

    # Compare results
    logger.info("\nCost comparison:")
    logger.info(f"Embedding join cost: ${embedding_cost:.6f}")
    logger.info(f"LLM join cost: ${llm_cost:.6f}")
    logger.info(f"Cheaper method: {'Embedding' if embedding_cost < llm_cost else 'LLM'} join")

    logger.info("\nAccuracy comparison:")
    logger.info(f"Embedding join accuracy: {embedding_metrics['accuracy']:.2%}")
    logger.info(f"LLM join accuracy: {llm_metrics['accuracy']:.2%}")
    if embedding_metrics['accuracy'] == llm_metrics['accuracy']:
        logger.info("Both methods have equal accuracy")
    else:
        logger.info(f"More accurate method: {'Embedding' if embedding_metrics['accuracy'] > llm_metrics['accuracy'] else 'LLM'} join")

if __name__ == "__main__":
    run_biodex_analysis()