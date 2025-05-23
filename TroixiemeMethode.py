import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

class SVDSearchEngineWithBidiagQR:
    def __init__(self, k=2, alpha=0.5, boost_factor=1.0):
        self.k = k
        self.alpha = alpha
        self.boost_factor = boost_factor

    def build_term_document_matrix(self, documents):
        self.doc_names = [doc[0] for doc in documents]
        vocabulary = sorted(set(term for _, terms in documents for term in terms))
        self.term_to_index = {term: i for i, term in enumerate(vocabulary)}
        self.index_to_term = {i: term for term, i in self.term_to_index.items()}

        self.matrix = np.zeros((len(vocabulary), len(documents)))
        for j, (_, terms) in enumerate(documents):
            for term in terms:
                if term in self.term_to_index:
                    self.matrix[self.term_to_index[term], j] += 1

    def compute_svd(self):
        A = self.matrix.copy()
        m, n = A.shape
        U_bi = np.eye(m)
        V_bi = np.eye(n)
        A_copy = A.copy()

        for i in range(min(m, n)):
            x = A_copy[i:, i]
            if norm(x) == 0:
                continue
            e1 = np.zeros_like(x)
            e1[0] = norm(x)
            u = x - e1
            u_norm = norm(u)
            if u_norm == 0:
                continue
            v = u / u_norm
            H = np.eye(m)
            H[i:, i:] -= 2.0 * np.outer(v, v)
            A_copy = H @ A_copy
            U_bi = U_bi @ H

            if i < n - 1:
                x = A_copy[i, i + 1:]
                if norm(x) == 0:
                    continue
                e1 = np.zeros_like(x)
                e1[0] = norm(x)
                u = x - e1
                u_norm = norm(u)
                if u_norm == 0:
                    continue
                v = u / u_norm
                H = np.eye(n)
                H[i + 1:, i + 1:] -= 2.0 * np.outer(v, v)
                A_copy = A_copy @ H
                V_bi = V_bi @ H

        # ⚠️ Fix: Do not truncate A_copy — keep full dimensions
        BTB = A_copy.T @ A_copy  # shape (n, n)

        eigvals, eigvecs = np.linalg.eigh(BTB)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        singular_values = np.sqrt(np.clip(eigvals, 0, None))

        # Fix: use the full eigvecs which now match n = A.shape[1]
        VT_b = eigvecs.T  # shape (n, n).T = (n, n) → we select [:k]

        VT_k = VT_b[:self.k, :]   # shape (k, n)
        AV = A @ VT_k.T           # shape (m, k) ← valid now

        S_inv = np.diag(1.0 / singular_values[:self.k])
        U_k = AV @ S_inv

        self.U = U_k
        self.S = singular_values[:self.k]
        self.VT = VT_k

        return self.U, self.S, self.VT



    def process_query(self, query_terms, threshold=0.5):
        if not hasattr(self, 'U') or not hasattr(self, 'S') or not hasattr(self, 'VT'):
            raise RuntimeError("SVD has not been computed.")

        q_vec = np.zeros((self.matrix.shape[0],))
        for term in query_terms:
            if term in self.term_to_index:
                q_vec[self.term_to_index[term]] += 1

        q_proj = self.U.T @ q_vec
        q_proj = q_proj[:self.k] / (norm(q_proj[:self.k]) + 1e-10)

        scores = {}
        for j in range(len(self.doc_names)):
            d_proj = self.VT[:, j]
            d_proj = d_proj[:self.k] / (norm(d_proj[:self.k]) + 1e-10)
            score = np.dot(q_proj, d_proj)
            scores[self.doc_names[j]] = score

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in ranked_docs if score > threshold]

# Testing Script
if __name__ == "__main__":
    documents = [
        ("Doc1", ["Croissance", "PIB", "Investissement"]),
        ("Doc2", ["Inflation", "Monnaie", "Dépression"]),
        ("Doc3", ["Commerce", "Exportation", "Croissance"]),
        ("Doc4", ["Emploi", "Chômage", "Salaires"]),
        ("Doc5", ["Impôts", "Fiscalité", "Revenu"]),
        ("Doc6", ["Géologie", "Faille", "Tremblement"]),
        ("Doc7", ["Volcan", "Sésme", "Plaque", "tectonique"]),
        ("Doc8", ["Dépression", "Bassin", "Érosion"]),
        ("Doc9", ["Stratigraphie", "Couches", "Roche"]),
        ("Doc10", ["Gisement", "Forage", "Bassin"])
    ]

    queries = {
        "q1": ["Dépression", "Bassin"],
        "q2": ["Croissance", "Fiscalité"]
    }

    k_values = [1, 2, 3, 5, 8, 10]

    engine = SVDSearchEngineWithBidiagQR(k=10)
    engine.build_term_document_matrix(documents)
    engine.compute_svd()

    print("Singular values (custom):", engine.S)

    U_np, S_np, VT_np = np.linalg.svd(engine.matrix, full_matrices=False)
    print("Numpy's singular values:", S_np[:10])
    print("Difference in singular values:", engine.S - S_np[:10])

    U_diff = norm(np.abs(engine.U) - np.abs(U_np[:, :10]))
    VT_diff = norm(np.abs(engine.VT) - np.abs(VT_np[:10, :]))
    print("Difference in U (up to sign):", U_diff)
    print("Difference in VT (up to sign):", VT_diff)

    for qname, qterms in queries.items():
        print(f"\nQuery {qname}: {qterms}")
        results = engine.process_query(qterms, threshold=0.3)
        for doc, score in results:
            print(f"  {doc}: {score:.4f}")
