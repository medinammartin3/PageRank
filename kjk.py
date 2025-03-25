import numpy as np

"""
Création d'une matrice de transition creuse (sparse) qui garde uniquement
les probabilités non nules du graphe (éléments non nuls de la matrice de 
transition classique). 
"""
def init_sparse_transition_matrix(graph):
    dim = len(graph)
    # Noeud: [transitions] --> transition = (destination, probabilité)
    matrix = {i: [] for i in range(1, dim+1)}
    for i, (node, links) in enumerate(graph.items()):
        if len(links) == 0:
            # État aborbant -> lien vers toutes les pages
            for j in range(dim):
                transition = (j, 1/dim)
                matrix[i+1].append(transition)
        else:
            proba = 1/len(links)
            for link in links:
                matrix[i+1].append((int(link), proba))
    return matrix



"""
Méthode de la puissance avec matrice creuse.
Multiplication matricielle réalisée en O(N).
"""
def power_method_sparse(matrix, dampingFactor, iterations=50, tolerance=1e-6):
    dim = len(matrix)
    pi = np.ones(dim) / dim  # Vecteur initial uniforme
    for _ in range(iterations):
        next_pi = np.zeros(dim)
        # Multiplication: Itérer uniquement sur les valeurs non nulles 
        #                 stockées dans la matrice creuse.
        for i, edges in matrix.items():
            for j, prob in edges:
                # Ajouter la contribution de la page actuelle
                next_pi[j-1] += dampingFactor * pi[i-1] * prob
        # Appliquer le facteur de téléportation à chaque élément de pi
        # (Avec une probabilité 1-dampingFactor, le surfeur saute vers une page aléatoire)
        teleportation = (1 - dampingFactor) / dim
        next_pi += teleportation
        # Vérifier la convergence du vecteur propre
        if np.linalg.norm(next_pi - pi, 1) < tolerance:
            break
        pi = next_pi
    return pi



"""
Algorithme PageRank optimisé les matrices creuses. 
Lequel est le cas du Web dans la réalité.
"""
def pageRank_sparse(graph, dampingFactor=0.85):
    # Créer la matrice sparse
    sparse_matrix = init_sparse_transition_matrix(graph)
    # Calculer le vecteur PageRank (probabilités stationnaires)
    pi = power_method_sparse(sparse_matrix, dampingFactor)
    return pi
