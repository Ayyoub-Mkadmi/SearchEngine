import time
import numpy as np
import random
from PremiereMethode import VectorSpaceModel
from DeuxiemeMethode import SVDSearchEngine
from TroixiemeMethode import SVDSearchEngineWithBidiagQR


def load_documents(file_path):
    """
    Charge le fichier documents.txt et extrait les données
    Chaque ligne contient le numéro du document suivi des mots-clés
    Exemple de format: '1 ECONOMIE PIB revenu chômage commerce'
    """
    documents = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # Supprimer les espaces inutiles
            if line:
                parts = line.split()
                doc_name = parts[0]  # Le premier élément est l'identifiant du document
                terms = parts[1:]    # Le reste sont les termes
                documents.append((doc_name, terms))  # Ajouter le document et ses termes à la liste
        
    return documents


def build_term_document_matrix(documents):
    """
    Construit la matrice termes-documents D et extrait les mots-clés T
    """
    all_terms = set()
    for _, terms in documents:
        all_terms.update(terms)  # Ajouter tous les termes uniques des documents
    
    terms = sorted(all_terms)  # Liste des termes (mots-clés) triés par ordre alphabétique
    documents_names = [doc_name for doc_name, _ in documents]
    
    num_terms = len(terms)
    num_docs = len(documents)
    
    # Initialisation de la matrice termes-documents (binaire)
    matrix = np.zeros((num_terms, num_docs))
    
    for doc_idx, (_, terms) in enumerate(documents):
        for term in terms:
            if term in terms:
                term_idx = terms.index(term)
                matrix[term_idx, doc_idx] = 1  # Remplir la matrice avec 1 si le terme est présent dans le document
    
    return matrix, terms, documents_names


def random_query(terms):
    """
    Sélectionne aléatoirement deux mots-clés parmi la liste des termes (T)
    """
    return random.sample(terms, 2)


def evaluate_methods_for_na(N, Na_values):
    """
    Évalue les méthodes pour différentes tailles de Na (nombre de documents)
    """
    times = []
    
    for Na in Na_values:
        # Créer un sous-ensemble de documents avec Na termes
        documents = load_documents("documents.txt")[:Na]
        matrix, terms, doc_names = build_term_document_matrix(documents)
        
        # Méthode 1 : VectorSpaceModel
        vsm = VectorSpaceModel()
        vsm.build_term_document_matrix(documents)
        query = random_query(terms)
        start_time = time.time()
        vsm.process_query(query, threshold=0.2)
        times.append(time.time() - start_time)

        # Méthode 2 : SVDSearchEngine
        svd_se = SVDSearchEngine(k=2)
        svd_se.build_term_document_matrix(documents)
        start_time = time.time()
        svd_se.process_query(query, threshold=0.2)
        times.append(time.time() - start_time)

        # Méthode 3 : SVDSearchEngineWithBidiagQR
        svd_bidiag_qr = SVDSearchEngineWithBidiagQR(k=2)
        svd_bidiag_qr.build_term_document_matrix(documents)
        start_time = time.time()
        svd_bidiag_qr.process_query(query, threshold=0.2)
        times.append(time.time() - start_time)
        
    return times


# Exécuter les différentes étapes
if __name__ == "__main__":
    # Charger les documents depuis le fichier "documents.txt"
    documents = load_documents("documents.txt")
    print(f"Nombre de documents chargés: {len(documents)}")

    # Construire la matrice termes-documents D et la liste des mots-clés T
    matrix, terms, documents_names = build_term_document_matrix(documents)

    # Afficher la matrice et les mots-clés
    print("Matrice termes-documents (D):")
    print(matrix)
    print("\nListe des mots-clés (T):")
    print(terms)

    # Choisir une requête aléatoire parmi les mots-clés (T)
    query = random_query(terms)
    print(f"\nRequête aléatoire choisie: {query}")

    # Utiliser VectorSpaceModel pour traiter la requête
    vsm = VectorSpaceModel()
    vsm.build_term_document_matrix(documents)
    results_vsm = vsm.process_query(query, threshold=0.2)
    print("\nRésultats pour la requête avec VectorSpaceModel:")
    for doc, score in results_vsm:
        print(f"- {doc}: score = {score:.2f}")

    # Utiliser SVDSearchEngine pour traiter la requête
    svd_se = SVDSearchEngine(k=2)
    svd_se.build_term_document_matrix(documents)
    results_svd_se = svd_se.process_query(query, threshold=0.2)
    print("\nRésultats pour la requête avec SVDSearchEngine:")
    for doc, score in results_svd_se:
        print(f"- {doc}: score = {score:.2f}")

    # Utiliser SVDSearchEngineWithBidiagQR pour traiter la requête
    svd_bidiag_qr = SVDSearchEngineWithBidiagQR(k=5)
    svd_bidiag_qr.build_term_document_matrix(documents)
    svd_bidiag_qr.compute_svd() 
    results_svd_bidiag_qr = svd_bidiag_qr.process_query(query, threshold=0.2)
    print("\nRésultats pour la requête avec SVDSearchEngineWithBidiagQR:")
    for doc, score in results_svd_bidiag_qr:
        print(f"- {doc}: score = {score:.2f}")

    print(f"Nombre de documents chargés: {len(documents)}")
