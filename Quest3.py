import numpy as np
import time
import matplotlib.pyplot as plt
from PremiereMethode import VectorSpaceModel
from DeuxiemeMethode import SVDSearchEngine
from TroixiemeMethode import SVDSearchEngineWithBidiagQR

# Generate documents and terms for the term-document matrix
def generate_documents(Na):
    terms = ["terme{}".format(i) for i in range(Na)]  # Create Na terms
    documents = []
    for i in range(Na):
        document = ["terme{}".format(np.random.randint(Na)) for _ in range(Na)]
        documents.append(("Doc{}".format(i + 1), document))
    return documents, terms

# Method to evaluate the execution time for each search engine method
def evaluate_methods_for_na(N, Na_values):
    times_vsm = []
    times_svd = []
    times_svd_bidiag_qr = []
    
    for Na in Na_values:
        documents, terms = generate_documents(Na)
        
        # VectorSpaceModel
        vsm = VectorSpaceModel()
        start_time = time.time()
        vsm.build_term_document_matrix(documents)
        query = np.random.choice(terms, 5, replace=False).tolist()
        vsm.process_query(query)
        times_vsm.append(time.time() - start_time)
        
        # SVDSearchEngine
        svd_se = SVDSearchEngine(k=2)
        start_time = time.time()
        svd_se.build_term_document_matrix(documents)
        svd_se.compute_svd()
        svd_se.process_query(query)
        times_svd.append(time.time() - start_time)
        
        # SVDSearchEngineWithBidiagQR
        svd_bidiag_qr = SVDSearchEngineWithBidiagQR(k=2, alpha=0.9, boost_factor=1.5)
        start_time = time.time()
        svd_bidiag_qr.build_term_document_matrix(documents)
        svd_bidiag_qr.compute_svd()
        svd_bidiag_qr.process_query(query)
        times_svd_bidiag_qr.append(time.time() - start_time)

    return times_vsm, times_svd, times_svd_bidiag_qr

# Main function to generate plots for execution times
def main():
    N = 300  # Fixed number of rows (terms)
    Na_values = [5, 10, 15, 20, 25, 50, 100, 150, 200]  # Different sizes of Na
    
    # Evaluate the methods for different sizes of Na
    times_vsm, times_svd, times_svd_bidiag_qr = evaluate_methods_for_na(N, Na_values)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    
    plt.plot(Na_values, times_vsm, label="VectorSpaceModel", marker='o', linestyle='-', color='b')
    plt.plot(Na_values, times_svd, label="SVDSearchEngine", marker='s', linestyle='-', color='g')
    plt.plot(Na_values, times_svd_bidiag_qr, label="SVD with Bidiagonalization", marker='^', linestyle='-', color='r')
    
    plt.xlabel("Number of Documents (Na)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time vs. Number of Documents (Na)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()


