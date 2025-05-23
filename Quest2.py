import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from PremiereMethode import VectorSpaceModel
from DeuxiemeMethode import SVDSearchEngine
from TroixiemeMethode import SVDSearchEngineWithBidiagQR

# Define the documents for Example 1
documents_ex1 = [
    ("D1", ["algèbre", "linéaire", "matrices"]),
    ("D2", ["analyse", "réelle", "suites"]),
    ("D3", ["probabilités", "statistiques"]),
    ("D4", ["matrices", "déterminants"])
]

queries_ex1 = {
    "q1": ["matrice", "algèbre"]
}

# Define the documents for Example 2
documents_ex2 = [
    ("Doc 1", ["Croissance", "PIB", "Investissement"]),
    ("Doc 2", ["Inflation", "Monnaie", "Dépression"]),
    ("Doc 3", ["Commerce", "Exportation", "Croissance"]),
    ("Doc 4", ["Emploi", "Chomage", "Salaires"]),
    ("Doc 5", ["Impots", "Fiscalité", "Revenu"]),
    ("Doc 6", ["Géologie", "Faille", "Tremblement"]),
    ("Doc 7", ["Volcan", "Séisme", "Plaque tectonique"]),
    ("Doc 8", ["Dépression", "Bassin", "Erosion"]),
    ("Doc 9", ["Stratigraphie", "Couches", "Roche"]),
    ("Doc 10", ["Gisement", "Forage", "Bassin"])
]

queries_ex2 = {
    "q2": ["Dépression", "Croissance"],
    "q3": ["Bassin", "Fiscalité"]
}

# Function to evaluate each method and plot results
def evaluate_method(method_class, documents, queries, method_name, k_values, thresholds):
    print(f"\n--- Method: {method_name} ---")
    reconstruction_errors = {}
    rankings = {}

    for k in k_values:
        print(f"\n>> k = {k}")
        engine = method_class(k=k) if method_name != "Basic" else method_class()
        engine.build_term_document_matrix(documents)

        if method_name != "Basic":
            engine.compute_svd()

            max_k = min(engine.U.shape[1], engine.VT.shape[0], len(engine.S))
            if k > max_k:
                print(f"Skipping k={k} (exceeds max rank)")

            # Your method
            U_k = engine.U[:, :k]
            S_k = np.diag(engine.S[:k])
            VT_k = engine.VT[:k, :]
            D_k = U_k @ S_k @ VT_k
            error = norm(engine.matrix - D_k, ord=2) / norm(engine.matrix, ord=2)

            # NumPy reference
            U_np, S_np, VT_np = np.linalg.svd(engine.matrix, full_matrices=False)
            D_k_ref = U_np[:, :k] @ np.diag(S_np[:k]) @ VT_np[:k, :]
            error_ref = norm(engine.matrix - D_k_ref, ord=2) / norm(engine.matrix, ord=2)

            reconstruction_errors[k] = error
            print(f"[Custom]     Spectral Error ||D - D_k||_2: {error:.6f}")
            print(f"[NumPy SVD] Spectral Error ||D - D_k||_2: {error_ref:.6f}")

        # Queries
        for query_name, query_terms in queries.items():
            print(f"\nQuery: {query_name} -> {query_terms}")
            query_rankings = []

            for threshold in thresholds:
                results = engine.process_query(query_terms, threshold=threshold)
                print(f"\n  Threshold > {threshold}")
                if results:
                    for doc, score in results:
                        print(f"    {doc}: {score:.3f}")
                else:
                    print("    No documents matched.")
                query_rankings.append((threshold, results))

            rankings[(query_name, k)] = query_rankings

    return reconstruction_errors, rankings

# Function to plot reconstruction errors
def plot_errors(error_dict, title):
    """
    Plot reconstruction error ||D - D_k||_2 as a function of k.
    """
    plt.figure(figsize=(8, 5))
    for method, errors in error_dict.items():
        k_vals = sorted(errors)
        err_vals = [errors[k] for k in k_vals]
        plt.plot(k_vals, err_vals, marker='o', label=method)
    plt.title(f"Spectral Norm Error vs k\n{title}")
    plt.xlabel("k")
    plt.ylabel("Spectral Norm Error ||D - D_k||_2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to compare rankings for a specific query and threshold across different k values
def compare_rankings(rankings, query_name, k_values, threshold=0.5):
    print(f"\n--- Comparing Rankings for Query: {query_name} (Threshold > {threshold}) ---")
    for k in k_values:
        key = (query_name, k)
        if key in rankings:
            print(f"\n>> k = {k}")
            for thresh, results in rankings[key]:
                if thresh == threshold:
                    print(f"  Results: {[(doc, f'{score:.3f}') for doc, score in results]}")

# Main function to run the evaluation
def main():
    methods = {
        "Basic": VectorSpaceModel,
        "SVD": SVDSearchEngine,
        "Bidiag+QR": SVDSearchEngineWithBidiagQR
    }

    k_values = [1, 2, 3, 5, 8, 10]  # Values of k to test
    thresholds = [0.3, 0.5, 0.7]  # Thresholds for filtering results

    # Example 1: Algebra & Matrices (Example 1)
    print("\n" + "="*50)
    print("  ANALYSIS: ALGEBRA & MATRICES DOCUMENTS (Example 1)")
    print("="*50)

    errors_ex1 = {}
    rankings_ex1 = {}
    for method_name, method_class in methods.items():
        errors, rankings = evaluate_method(method_class, documents_ex1, queries_ex1, method_name, k_values, thresholds)
        if errors:
            errors_ex1[method_name] = errors
        if rankings:
            rankings_ex1.update(rankings)

    plot_errors(errors_ex1, "Algebra & Matrices Documents (Example 1)")

    # Example 2: Economics & Geology (Example 2)
    print("\n" + "="*50)
    print("  ANALYSIS: ECONOMICS & GEOLOGY DOCUMENTS (Example 2)")
    print("="*50)

    errors_ex2 = {}
    rankings_ex2 = {}
    for method_name, method_class in methods.items():
        errors, rankings = evaluate_method(method_class, documents_ex2, queries_ex2, method_name, k_values, thresholds)
        if errors:
            errors_ex2[method_name] = errors
        if rankings:
            rankings_ex2.update(rankings)

    plot_errors(errors_ex2, "Economics & Geology Documents (Example 2)")

    # Comparing rankings for a specific query
    compare_rankings(rankings_ex2, "q2", k_values, threshold=0.5)
    compare_rankings(rankings_ex2, "q3", k_values, threshold=0.5)

if __name__ == "__main__":
    main()
