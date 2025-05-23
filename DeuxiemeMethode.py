import numpy as np
from numpy.linalg import norm

class SVDSearchEngine:
    def __init__(self, k=2):
        self.terms = []
        self.documents = []
        self.matrix = None
        self.k = k
        self.U = None
        self.S = None
        self.VT = None

    def build_term_document_matrix(self, documents):
        all_terms = set()
        for doc_name, terms in documents:
            all_terms.update(terms)

        self.terms = sorted(all_terms)
        self.documents = [doc_name for doc_name, _ in documents]

        num_terms = len(self.terms)
        num_docs = len(documents)
        self.matrix = np.zeros((num_terms, num_docs))

        for doc_idx, (_, terms) in enumerate(documents):
            for term in terms:
                if term in self.terms:
                    term_idx = self.terms.index(term)
                    self.matrix[term_idx, doc_idx] = 1

    def power_iteration(self, A, num_iterations=100):
        n = A.shape[1]
        v = np.random.rand(n)
        v = v / norm(v)

        for _ in range(num_iterations):
            Av = A @ v
            v_new = Av / norm(Av)
            if np.allclose(v, v_new, atol=1e-10):
                break
            v = v_new

        eigenvalue = norm(A @ v)
        return eigenvalue, v

    def compute_svd(self):
        if self.matrix is None:
            raise ValueError("Term-document matrix not built")

        D = self.matrix
        A = D.T @ D

        eigenvalues = []
        eigenvectors = []

        for _ in range(self.k):
            if eigenvalues:
                B = A - sum(s * np.outer(v, v) for s, v in zip(eigenvalues, eigenvectors))
                sigma, v = self.power_iteration(B)
            else:
                sigma, v = self.power_iteration(A)

            eigenvalues.append(sigma)
            eigenvectors.append(v)

        # Sort in descending order of singular values
        S = np.sqrt(np.array(eigenvalues))
        V = np.column_stack(eigenvectors)
        order = np.argsort(S)[::-1]

        self.S = S[order]
        V = V[:, order]
        self.VT = V.T
        self.U = D @ V @ np.diag(1 / self.S)

        return self.U, np.diag(self.S), self.VT

    def reduced_rank_approximation(self):
        if self.U is None or self.S is None or self.VT is None:
            self.compute_svd()

        U_k = self.U[:, :self.k]
        S_k = np.diag(self.S[:self.k])
        VT_k = self.VT[:self.k, :]

        return U_k @ S_k @ VT_k

    def process_query(self, query_terms, threshold=0.8):
        if self.U is None or self.S is None or self.VT is None:
            self.compute_svd()

        q = np.zeros(len(self.terms))
        for term in query_terms:
            if term in self.terms:
                term_idx = self.terms.index(term)
                q[term_idx] = 1

        U_k = self.U[:, :self.k]
        S_k = self.S[:self.k]
        VT_k = self.VT[:self.k, :]

        q_k = U_k.T @ q

        scores = []
        for doc_idx in range(len(self.documents)):
            d_k = S_k * VT_k[:, doc_idx]
            dot_product = np.dot(q_k, d_k)
            q_norm = norm(q_k)
            d_norm = norm(d_k)
            score = dot_product / (q_norm * d_norm) if q_norm != 0 and d_norm != 0 else 0
            scores.append((self.documents[doc_idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in scores if score >= threshold]

if __name__ == "__main__":
    documents = [
        ("D1", ["algèbre", "linéaire", "matrices"]),
        ("D2", ["analyse", "réelle", "suites"]),
        ("D3", ["probabilités", "statistiques"]),
        ("D4", ["matrices", "déterminants"])
    ]

    svd_se = SVDSearchEngine(k=2)
    svd_se.build_term_document_matrix(documents)
    U, S, VT = svd_se.compute_svd()

    print("Matrice originale D:")
    print(svd_se.matrix)
    print("\nApproximation D_k:")
    print(svd_se.reduced_rank_approximation())

    query = ["matrices", "algèbre"]
    results = svd_se.process_query(query, threshold=0.5)

    print("\nRésultats pour la requête:", query)
    for doc, score in results:
        print(f"- {doc}: score = {score:.2f}")
