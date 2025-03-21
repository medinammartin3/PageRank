from pagerank import generate_graph
import numpy as np
import matplotlib.pyplot as plt
import time

"""
O(k * (N + E)) just for sparse matrix
"""

def init_sparse_transition_matrix(graph):
    """Création d'une matrice de transition sparse."""
    dim = len(graph)
    matrix = {i: [] for i in range(dim)}

    for i, (node, links) in enumerate(graph.items()):
        if len(links) == 0:
            # Dangling node -> link to all pages
            for j in range(dim):
                matrix[i].append((j, 1 / dim))
        else:
            prob = 1 / len(links)
            for link in links:
                matrix[i].append((int(link) - 1, prob))
    
    return matrix


def power_method_sparse(matrix, dampingFactor, dim, iterations=50, tolerance=1e-6):
    """Méthode de la puissance avec matrice sparse."""
    pi = np.ones(dim) / dim  # Vecteur initial uniforme

    for _ in range(iterations):
        next_pi = np.zeros(dim)

        # Sparse matrix-vector multiplication
        for i, edges in matrix.items():
            for j, prob in edges:
                next_pi[j] += dampingFactor * pi[i] * prob

        # Add teleportation factor
        teleportation = (1 - dampingFactor) / dim
        next_pi += teleportation

        # Check for convergence
        if np.linalg.norm(next_pi - pi, 1) < tolerance:
            break

        pi = next_pi

    return pi


def pageRank_sparse(graph, dampingFactor=0.85):
    """Algorithme PageRank optimisé avec matrice sparse."""
    dim = len(graph)

    # Créer la matrice sparse
    sparse_matrix = init_sparse_transition_matrix(graph)

    # Calculer le vecteur PageRank
    pi = power_method_sparse(sparse_matrix, dampingFactor, dim)

    # print("PageRank:", pi)
    # print("Sum:", pi.sum())
    return pi

# www = {"1": [2,3], "2": [], "3": [1,2,5], "4": [5,6], "5": [4,6], "6":[4]}
# pageRank_sparse(www, dampingFactor=0.9)

