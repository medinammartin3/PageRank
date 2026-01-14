import numpy as np 
import random


"""
Complexité: O(k * (N^2 + E))
"""


"""
Génération de la matrice des probabilités de transition d'un graphe donné.
La matrice retournée est stochastique.

Paramètres:
- graph: Graphe représentant le Web.
"""
def initTransitionsMatrix(graph):
    dim = len(graph)  # Nombre de noeuds (pages web)
    matrix = np.zeros((dim, dim), dtype=np.float64)  # Matrice des probabilités de transition

    for i in range(len(graph)):
        links = graph[str(i+1)]  # Liens sortant de la page
        line = matrix[i]  # Ligne correspondante à la page dsns la matrice

        # Ajout d'un lien vers toute les pages si 
        # la page actuelle n'a pas de liens sortants
        if len(links) == 0:
            line += 1/dim
        else: 
            # Insertion des transitions dans la matrice
            for link in links:
                line[link-1] = 1 / len(links)  # Normalisation des probabilités

    return matrix



"""
Génération d'un graphe aléatoire de taille donnée.

Paramètres:
- num_nodes: Nombre de noeuds (pages Web).
- max_links: Nombre maximal de liens que un noeud (page) accepte.
"""
def generate_random_graph(num_nodes, max_links):
    graph = {}
    for node in range(1, num_nodes + 1):
        # Nombre et destination aléatoire des liens
        num_links = random.randint(0, max_links)  
        links = random.sample(range(1, num_nodes + 1), num_links) 
        graph[str(node)] = links
    return graph


"""
Génération d'un graphe de taille et acrdinalité de cardinalité de chaque noeud donnée.

Paramètres:
- num_nodes: Nombre de noeuds (pages Web).
- num_links: Nombre de liens de chaque noeud (page).
"""
def generate_graph(num_nodes, num_links):
    graph = {}
    for node in range(1, num_nodes + 1):
        # Destination aléatoire des liens
        links = random.sample(range(1, num_nodes + 1), num_links) 
        graph[str(node)] = links
    return graph



"""
Méthode de la puissance pour calculer le vecteur des probabilités d'équilibre 
(stationnaires) d'une matrice des probabilités de transition.

Paramètres:
- transitionsMatrix: Matrice des probabilités de transition.
- iterations: Nombre d'itérations maximal.
- tolerance: Seuil de tolérance pour assumer la convergence.
"""
def powerMethod(transitionsMatrix, iterations=50, tolerance=1e-6):
    P = transitionsMatrix
    dim = len(P)
    pi = np.ones(dim) * 1/dim  # Initialisation uniforme du vecteur des probabilités d'équilibre
    # lam = 0
    
    for _ in range(iterations):
        pi_next = pi @ P  # Nouveau vecteur calculé (vecteur propre)

        # Vérification de varialition significative (convergence) du vecteur propre.
        if np.linalg.norm(pi_next - pi) < tolerance:
            break

        pi = pi_next  # Actualisation du vecteur des probabilités d'équilibre

    return pi



"""
Algorithme PageRank qui donne le score de rélévance de chaque page d'un graphe.

Paremètres:
- graph: Graphe représentant le Web.
- dampingFactor: Facteur d'amortissement (probabilité de téléportement).
"""
def pageRank(graph, dampingFactor=0.85):
    transitionsMatrix = initTransitionsMatrix(graph)
    dim = len(transitionsMatrix)
    eeT = np.ones((dim, dim)) * 1/dim

    # Rendre irréductible la matrice des probabilités de transition
    P = dampingFactor * transitionsMatrix + (1-dampingFactor) * eeT
    print(P)

    # Calcul du vecteur des probabilités d'équilibre (stationnaires)
    pi = powerMethod(P, iterations=50, tolerance=1e-6)

    print("PageRank:", pi)
    print("Sum:", pi.sum())
    return pi

# graphExample = {"1": [2,3], "2": [3,4,5], "3": [1,2], "4": [3], "5": []}
# pageRank(www, dampingFactor=0.85)