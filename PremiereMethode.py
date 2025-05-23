import numpy as np

class VectorSpaceModel:
    def __init__(self):
        """Initialise le modèle d'espace vectoriel"""
        self.terms = []  # Liste des termes
        self.documents = []  # Noms des documents
        self.matrix = None  # Matrice termes-documents
    
    def build_term_document_matrix(self, documents):
        """
        Construit la matrice termes-documents à partir d'une liste de documents
        
        Args:
            documents: Liste de tuples (nom_du_document, liste_des_termes)
        """
        # Extraire tous les termes uniques et les trier par ordre alphabétique
        all_terms = set()
        for doc_name, terms in documents:
            all_terms.update(terms)
        
        self.terms = sorted(all_terms)
        self.documents = [doc_name for doc_name, _ in documents]
        
        # Construire la matrice termes-documents (binaire)
        num_terms = len(self.terms)
        num_docs = len(documents)
        self.matrix = np.zeros((num_terms, num_docs))
        
        for doc_idx, (_, terms) in enumerate(documents):
            for term in terms:
                if term in self.terms:
                    term_idx = self.terms.index(term)
                    self.matrix[term_idx, doc_idx] = 1
    
    def process_query(self, query_terms, threshold=0.8):
        """
        Traite une requête et retourne les documents pertinents
        
        Args:
            query_terms: Liste des termes de la requête
            threshold: Seuil de pertinence (défaut: 0.8)
            
        Returns:
            Liste de tuples (document, score) triés par score décroissant
        """
        if self.matrix is None:
            raise ValueError("La matrice termes-documents n'a pas été construite")
        
        # Créer le vecteur requête
        query_vec = np.zeros(len(self.terms))
        for term in query_terms:
            if term in self.terms:
                term_idx = self.terms.index(term)
                query_vec[term_idx] = 1
        
        # Calculer les scores de similarité cosinus
        scores = []
        for doc_idx in range(len(self.documents)):
            doc_vec = self.matrix[:, doc_idx]
            
            # Produit scalaire
            dot_product = np.dot(query_vec, doc_vec)
            
            # Normes
            query_norm = np.linalg.norm(query_vec)
            doc_norm = np.linalg.norm(doc_vec)
            
            # Éviter la division par zéro
            if query_norm == 0 or doc_norm == 0:
                score = 0
            else:
                score = dot_product / (query_norm * doc_norm)
            
            scores.append((self.documents[doc_idx], score))
        
        # Trier par score décroissant et filtrer par seuil
        scores.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = [(doc, score) for doc, score in scores if score >= threshold]
        
        return relevant_docs
    




# Exemple d'utilisation avec les données fournies
if __name__ == "__main__":
    # Documents de l'exemple 1
    # documents = [
    #     ("D1", ["algèbre",  "matrices"]),
    #     ("D2", ["analyse", "réelle", "suites"]),
    #     ("D3", ["probabilités", "statistiques"]),
    #     ("D4", ["matrices", "déterminants"])
    # ]
    documents = [
        ("Doc 1", ["Croissance", "PIB", "Investissement"]),
        ("Doc 2", ["Inflation", "Monnaie", "Dépression"]),
        ("Doc 3", ["Commerce", "Exportation", "Croissance"]),
        ("Doc 4", ["Emploi", "Chomage", "Salaires"]),
        ("Doc 5", ["Impots", "Fiscalité", "Revenu"]),
        ("Doc 6", ["Géologie", "Faille", "Tremblement"]),
        ("Doc 7", ["Volcan", "Séisme", "Plaque tectonique"]),
        ("Doc 8", ["Dépression", "Bassin", "Erosion"]),
        ("Doc 9", ["Stratigraphie", "Couches", "Roche"]),
        ("Doc 10", ["Gisement", "Forage", "Bassin"]),
    ]

    
    # Construire le modèle
    vsm = VectorSpaceModel()
    vsm.build_term_document_matrix(documents)
    
    # Requête de l'exemple 2
    query1 = ["Dépression", "Croissance"]

    query = ["Bassin", "Fiscalité"]
    
    # Calculer les scores
    results = vsm.process_query(query, threshold=0.2)
    
    # Afficher les résultats
    print("Termes dans le modèle:", vsm.terms)
    print("\nMatrice termes-documents:")
    print(vsm.matrix)
    print("\nRésultats pour la requête:", query)
    for doc, score in results:
        print(f"- {doc}: score = {score:.2f}")