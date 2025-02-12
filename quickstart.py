from src.data_loader import DataLoader
from src.semantic_join import SemanticJoin
import pandas as pd

def run_benchmark():
    print("Starting benchmark test...")

    print("Creating data for DataFrame A...")
    data_a = pd.DataFrame({
        'description': [
            'Introduction to Programming',
            'Advanced Mathematics',
            'Database Systems'
        ]
    })
    print("DataFrame A created:", data_a)

    print("Creating data for DataFrame B...")
    data_b = pd.DataFrame({
        'skill': [
            'Python Programming',
            'Statistical Analysis',
            'SQL'
        ]
    })
    print("DataFrame B created:", data_b)

    # Ground truth for accuracy measurement
    ground_truth = [
        ('Introduction to Programming', 'Python Programming'),
        ('Advanced Mathematics', 'Statistical Analysis'),
        ('Database Systems', 'SQL')
    ]
    print("Ground truth data:", ground_truth)

    # Initialize semantic join
    print("Initializing SemanticJoin...")
    join_op = SemanticJoin(data_loader=None)  # Replace with actual data loader if needed
    print("SemanticJoin initialized.")

    # Benchmark embedding join
    print("Benchmarking embedding join...")
    embedding_metrics = join_op.benchmark(
        join_method='embedding',
        data_a=data_a, 
        data_b=data_b,
        truth=ground_truth,
        desc_col='description',
        label_col='skill'
    )
    embedding_cost = embedding_metrics['cost']
    print("Embedding Join Metrics:", embedding_metrics)

    # Benchmark LLM join 
    print("\nBenchmarking LLM join...")
    llm_metrics = join_op.benchmark(
        join_method='llm',
        data_a=data_a,
        data_b=data_b, 
        truth=ground_truth,
        col_a='description',
        col_b='skill'
    )
    llm_cost = llm_metrics['cost']
    print("LLM Join Metrics:", llm_metrics)

    print("\nCost comparison:")
    print(f"Embedding join cost: ${embedding_cost:.6f}")
    print(f"LLM join cost: ${llm_cost:.6f}")
    print(f"Cheaper method: {'Embedding' if embedding_cost < llm_cost else 'LLM'} join")

    print("\nAccuracy comparison:")
    print(f"Embedding join accuracy: {embedding_metrics['accuracy']:.2%}")
    print(f"LLM join accuracy: {llm_metrics['accuracy']:.2%}")
    if embedding_metrics['accuracy'] == llm_metrics['accuracy']:
        print("Both methods have equal accuracy")
    else:
        print(f"More accurate method: {'Embedding' if embedding_metrics['accuracy'] > llm_metrics['accuracy'] else 'LLM'} join")

if __name__ == "__main__":
    run_benchmark()