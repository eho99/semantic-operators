# from src.data_loader import DataLoader
# from src.semantic_join import SemanticJoin
# import pandas as pd

# def run_benchmark():
#     print("Starting benchmark test...")

#     print("Creating data for DataFrame A...")
#     data_a = pd.DataFrame({
#         'id': ['A1', 'A2', 'A3'],
#         'description': [
#             'Introduction to Programming',
#             'Advanced Mathematics',
#             'Database Systems'
#         ]
#     })
#     print("DataFrame A created:", data_a)

#     print("Creating data for DataFrame B...")
#     data_b = pd.DataFrame({
#         'id': ['B1', 'B2', 'B3'],
#         'skill': [
#             'Python Programming',
#             'Statistical Analysis',
#             'SQL'
#         ]
#     })
#     print("DataFrame B created:", data_b)

#     # Ground truth for accuracy measurement, using IDs
#     ground_truth = [
#         ('A1', 'B1'),
#         ('A2', 'B2'),
#         ('A3', 'B3')
#     ]
#     print("Ground truth data:", ground_truth)

#     # Initialize semantic join
#     print("Initializing SemanticJoin...")
#     join_op = SemanticJoin(data_loader=None)  # Replace with actual data loader if needed
#     print("SemanticJoin initialized.")

#     # Benchmark embedding join
#     print("Benchmarking embedding join...")
#     embedding_metrics = join_op.benchmark(
#         join_method='embedding',
#         data_a=data_a, 
#         data_b=data_b,
#         truth=ground_truth,
#         id_col='id',
#         desc_col='description',
#         label_col='skill'
#     )
#     embedding_cost = embedding_metrics['cost']
#     print("Embedding Join Metrics:", embedding_metrics)

#     # Benchmark LLM join 
#     print("\nBenchmarking LLM join...")
#     llm_metrics = join_op.benchmark(
#         join_method='llm',
#         data_a=data_a,
#         data_b=data_b, 
#         truth=ground_truth,
#         id_col='id',
#         col_a='description',
#         col_b='skill'
#     )
#     llm_cost = llm_metrics['cost']
#     print("LLM Join Metrics:", llm_metrics)

#     print("\nCost comparison:")
#     print(f"Embedding join cost: ${embedding_cost:.6f}")
#     print(f"LLM join cost: ${llm_cost:.6f}")
#     print(f"Cheaper method: {'Embedding' if embedding_cost < llm_cost else 'LLM'} join")

#     print("\nAccuracy comparison:")
#     print(f"Embedding join accuracy: {embedding_metrics['accuracy']:.2%}")
#     print(f"LLM join accuracy: {llm_metrics['accuracy']:.2%}")
#     if embedding_metrics['accuracy'] == llm_metrics['accuracy']:
#         print("Both methods have equal accuracy")
#     else:
#         print(f"More accurate method: {'Embedding' if embedding_metrics['accuracy'] > llm_metrics['accuracy'] else 'LLM'} join")

# if __name__ == "__main__":
#     run_benchmark()

from src.data_loader import DataLoader
from src.semantic_join import SemanticJoin
import pandas as pd

def run_benchmark():
    print("Starting benchmark test...")

    print("Creating data for DataFrame A...")
    data_a = pd.DataFrame({
        'id': ['A1', 'A2', 'A3'],
        'description': [
            'Introduction to Programming',
            'Advanced Mathematics',
            'Database Systems'
        ]
    })
    print("DataFrame A created:\n", data_a)

    print("Creating data for DataFrame B...")
    data_b = pd.DataFrame({
        'id': ['B1', 'B2', 'B3'],
        'skill': [
            'Python Programming',
            'Statistical Analysis',
            'SQL'
        ]
    })
    print("DataFrame B created:\n", data_b)

    # Ground truth for accuracy measurement, using IDs.
    # For the embedding join we expect the returned DataFrame to have columns "sample_id" and "matched_label"
    ground_truth_embedding = pd.DataFrame( 
        [('A1', 'Python Programming'), ('A2', 'Statistical Analysis'), ('A3', 'SQL')],
        columns=["sample_id", "matched_label"]
    )
    print("Embedding Ground truth data:\n", ground_truth_embedding)

    # For the LLM join we expect the returned DataFrame to have columns "sample_id" and "value_b"
    ground_truth_llm = pd.DataFrame( 
        [('A1', 'B1'), ('A2', 'B2'), ('A3', 'B3')],
        columns=["sample_id", "value_b"]
    )
    print("LLM Ground truth data:\n", ground_truth_llm)

    # Initialize semantic join (data_loader is not needed in this quickstart)
    print("Initializing SemanticJoin...")
    join_op = SemanticJoin(data_loader=None)
    print("SemanticJoin initialized.")

    # Benchmark embedding join
    print("Benchmarking embedding join...")
    embedding_metrics = join_op.benchmark(
        join_method='embedding',
        data_a=data_a, 
        data_b=data_b,
        truth=ground_truth_embedding,
        id_col='id',
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
        truth=ground_truth_llm,
        id_col='id',
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