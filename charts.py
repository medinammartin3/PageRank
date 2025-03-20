from pagerank import generate_random_graph, generate_graph, pageRank, initTransitionsMatrix, powerMethod
import matplotlib.pyplot as plt
import numpy as np
import time
import math


"""
Analyse du ritme de convergence et de la réduction de l'erreur à chaque itération
de la méthode de la puissance.

Montre que même si les variations entre les valeurs du vecteur propre ncontinuent à
descendre (le vecteur propre n'a pas encore convergé), l'erreur entre le vrai
vecteur propre et celui approximé converge bien avant et ne continue pas à diminuer
avec des variations très petites. Pour cette raison, le seuil de tolérance de 1e-6
pour accepter la convergence est accepté.
"""
def powerMethod_Analysis1(transitionsMatrix, iterations=50, tolerance=1e-6):
    P = transitionsMatrix
    dim = len(P)
    pi = np.ones(dim) * 1/dim
    real_pi = np.array([0.03721, 0.05396, 0.04151, 0.3751, 0.206, 0.2862]) 
    # eignenvalue = 0

    errors = []  # Erreur à chaque itération
    variations = []  # Variation entre chaque itération
    values = [[] for _ in range(dim)]  # Valeur de chaque composante de pi à chaque itération

    for _ in range(iterations):
        pi_next = pi @ P

        # eignenvalue_next = (pi.T @ P @ pi) / (pi.T @ pi) 

        variation = np.linalg.norm(pi_next - pi)
        variations.append(variation)

        error = np.linalg.norm(pi_next - real_pi)
        errors.append(error)

        for i in range(dim):
            values[i].append(pi[i])

        if np.linalg.norm(pi_next - pi) < tolerance:
            break

        pi = pi_next
        # eignenvalue = eignenvalue_next

    return pi, errors, variations, values

def pageRank_Analysis1(graph, dampingFactor):
    transitionsMatrix = initTransitionsMatrix(graph)
    dim = len(transitionsMatrix)
    eeT = np.ones((dim, dim)) * 1/dim
    P = dampingFactor * transitionsMatrix + (1-dampingFactor) * eeT
    pi, errors, variations, _ = powerMethod_Analysis1(P, iterations=100, tolerance=0)
    return pi, errors, variations


www = {"1": [2,3], "2": [], "3": [1,2,5], "4": [5,6], "5": [4,6], "6":[4]}
iterations = 50
damping = 0.9
pi, errors, variations = pageRank_Analysis1(www, damping)

def chart1():
    plt.figure(figsize=(8, 5))
    plt.plot(errors, linestyle='-', color='b')
    plt.yscale('log')  # Using log scale to better visualize error decay
    plt.title('Convergence of Power Method - Error vs. Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Error (Log Scale)')
    plt.grid(True)
    plt.show()

def chart2():
    plt.figure(figsize=(8, 5))
    plt.plot(variations, linestyle='-', color='r')
    plt.yscale('log')  # Using log scale to better visualize error decay
    plt.title('Convergence of Power Method - Variation vs. Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Variation')
    plt.grid(True)
    plt.show()


""""
Convergence individuelle de chaque valeur du vecteur des probabilités d'équilibre.
"""

def pageRank_Analysis2(graph, dampingFactor=0.85):
    transitionsMatrix = initTransitionsMatrix(graph)
    dim = len(transitionsMatrix)
    eeT = np.ones((dim, dim)) * 1/dim
    P = dampingFactor * transitionsMatrix + (1-dampingFactor) * eeT
    _, _, _, values = powerMethod_Analysis1(P, iterations=100, tolerance=1e-6)
    return values

def chart3():
    elements_values = pageRank_Analysis2(www, damping)

    plt.figure(figsize=(10, 6))
    for i, values in enumerate(elements_values):
        x = np.arange(len(values))
        plt.plot(x, values, label=f'$\\pi_{{{i+1}}}$')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Convergence de chaque valeur')
    plt.legend()
    plt.grid(True)
    plt.show()



"""
Temps d'exécution de PageRank en fonction du nombre de noeuds, 
ce qui revient à la taille du graphe car les liens augmentent 
aussi au fûr et à mesure que le nombre de noeuds augmente.
"""

def chart4():
    pageRankTime = []
    num_nodes = []

    for i in range(5, 1000, 10):
        num_nodes.append(i)
        graph = generate_graph(i, min(10, math.ceil(i/2)))
        pageRank_start = time.time()
        pageRank(graph)
        pageRank_end = time.time()
        total_time = pageRank_end - pageRank_start
        pageRankTime.append(total_time)

    plt.figure(figsize=(8, 5))
    plt.plot(num_nodes, pageRankTime, linestyle='-', color='y')
    plt.title('Time complexity PageRank - Nodes')
    plt.xlabel('Number of nodes')
    plt.ylabel('Time')
    plt.grid(True)
    plt.show()


"""
Temps d'exécution de pagerank en fonction du nombre de liens (arrêtes du graphe).
"""

def chart5():
    pageRankTime = []
    num_edges = []

    for i in range(50, 400, 5):
        num_edges.append(i*500)
        graph = generate_graph(500, i)
        pageRank_start = time.time()
        for _ in range(5):
            pageRank(graph)
        pageRank_end = time.time()
        total_time = (pageRank_end - pageRank_start)/5
        pageRankTime.append(total_time)
    plt.figure(figsize=(8, 5))
    plt.plot(num_edges, pageRankTime, linestyle='-', color='y')
    plt.title('Time complexity PageRank - Edges')
    plt.xlabel('Number of Edges')
    plt.ylabel('Time')
    plt.grid(True)
    plt.show()


"""
Temps d'exécution de Power Method en fonction de la dimension de la matrice.
"""

def pageRank_Analysis3(graph, dampingFactor=0.85):
    transitionsMatrix = initTransitionsMatrix(graph)
    dim = len(transitionsMatrix)
    eeT = np.ones((dim, dim)) * 1/dim
    P = dampingFactor * transitionsMatrix + (1-dampingFactor) * eeT
    powerMethod_start = time.time()
    for _ in range(5):
        pi = powerMethod(P, iterations=100, tolerance=1e-6)
    powerMethod_end = time.time()
    total_time = (powerMethod_end - powerMethod_start) / 5
    return pi, total_time

def chart6():
    powerMethodTime = []
    num_nodes = []

    for i in range(5, 1000, 10):
        num_nodes.append(i)
        graph = generate_random_graph(i, math.ceil(i/3))
        _, total_time = pageRank_Analysis3(graph)
        powerMethodTime.append(total_time)

    plt.figure(figsize=(8, 5))
    plt.plot(num_nodes, powerMethodTime, linestyle='-', color='y')
    plt.title('Time complexity Power Method - Nodes')
    plt.xlabel('Number of nodes')
    plt.ylabel('Time')
    plt.grid(True)
    plt.show()


"""
Taux de convergence de Power Method en fonction de la dimension de la matrice.

Montre que le taux de convergence est indépendant de la taille de la matrice.
"""

def powerMethod_Analysis4(transitionsMatrix, iterations=50, tolerance=1e-6):
    P = transitionsMatrix
    dim = len(P)
    pi = np.ones(dim) * 1/dim
    for i in range(iterations):
        pi_next = pi @ P
        if np.linalg.norm(pi_next - pi) < tolerance:
            convergence_iter = i+1
            break
        pi = pi_next
    return convergence_iter

def pageRank_Analysis4(graph, dampingFactor=0.85):
    transitionsMatrix = initTransitionsMatrix(graph)
    dim = len(transitionsMatrix)
    eeT = np.ones((dim, dim)) * 1/dim
    P = dampingFactor * transitionsMatrix + (1-dampingFactor) * eeT
    convergence_iter = powerMethod_Analysis4(P, iterations=100, tolerance=1e-6)
    return convergence_iter

def chart7():
    num_nodes = []
    convergence_iters = []

    for i in range(10, 2000, 10):
        num_nodes.append(i)
        graph = generate_graph(i, min(10, math.ceil(i/2)))
        convergence_iter = pageRank_Analysis4(graph)
        convergence_iters.append(convergence_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(num_nodes, convergence_iters, linestyle='-', color='y')
    plt.title('Convergence rate Power Method - Nodes')
    plt.xlabel('Number of nodes')
    plt.ylim(0, 15)
    plt.ylabel('Iteration')
    plt.grid(True)
    plt.show()



"""
Taux de convergence de Power Method en fonction du damping factor.

Montre que le taux de convergence est dépendant ddu damping factor.
"""

def chart8():
    convergence_iters = []
    dampingFactors = []
    graph = generate_random_graph(500, 30)

    for i in range(0, 1000, 1):
        damping = i/1000
        dampingFactors.append(damping)
        convergence_iter = pageRank_Analysis4(graph, dampingFactor=damping)
        convergence_iters.append(convergence_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(dampingFactors, convergence_iters, linestyle='-', color='y')
    plt.title('Convergence rate - Damping')
    plt.ylabel('Iteration')
    plt.xlabel('Damping')
    plt.grid(True)
    plt.show()


## GÉNÉRATION DES GRAPHIQUES ##

# chart1()
# chart2()
# chart3()
# chart4()
# chart5()
# chart6()
# chart7()
# chart8()