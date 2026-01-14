from pagerank import generate_random_graph, generate_graph, pageRank, initTransitionsMatrix, powerMethod
from pagerank_sparse import pageRank_sparse
import matplotlib.pyplot as plt
import numpy as np
import time
import math


"""
Analyse du ritme de convergence et de la réduction de l'erreur à chaque itération
de la méthode de la puissance.

Montre que même si les variations entre les valeurs du vecteur propre continuent à
descendre (le vecteur propre n'a pas encore convergé), l'erreur entre le vrai
vecteur propre et celui approximé converge bien avant et ne continue pas à diminuer
avec des variations très petites. Pour cette raison, le seuil de tolérance de 1e-6
pour accepter la convergence est accepté.
On fait la convergence par rapport au vecteur propre et non à la valeur propre
puisque la convergence du vecteur propre est plus rapide et c'est celle qui nous 
intéresse. Lorsqu'on fait la convergence avec la valeur propre, le résultat est
le même.
"""
def powerMethod_Analysis1(transitionsMatrix, iterations=50, tolerance=1e-6):
    P = transitionsMatrix
    dim = len(P)
    pi = np.ones(dim) * 1/dim
    real_pi = np.array([12,16,9,1,3]) * (1/41)
    print(real_pi)
    errors = []  # Erreur à chaque itération
    variations = []  # Variation entre chaque itération
    values = [[] for _ in range(dim)]  # Valeur de chaque composante de pi à chaque itération

    for _ in range(iterations):
        pi_next = pi @ P

        variation = np.linalg.norm(pi_next - pi)
        variations.append(variation)

        error = np.linalg.norm(pi_next - real_pi)
        errors.append(error)

        for i in range(dim):
            values[i].append(pi[i])

        if np.linalg.norm(pi_next - pi) < tolerance:
            break

        pi = pi_next
    
    return pi, errors, variations, values

def pageRank_Analysis1(graph, dampingFactor):
    transitionsMatrix = initTransitionsMatrix(graph)
    dim = len(transitionsMatrix)
    eeT = np.ones((dim, dim)) * 1/dim
    P = dampingFactor * transitionsMatrix + (1-dampingFactor) * eeT
    pi, errors, variations, _ = powerMethod_Analysis1(P, iterations=100, tolerance=0)
    print(pi)
    return pi, errors, variations


# www = {"1": [2,3], "2": [], "3": [1,2,5], "4": [5,6], "5": [4,6], "6":[4]}
www = {"1": [2], "2": [1,3], "3": [1,2,5], "4": [1], "5": [2,4,3]}
damping = 0.9
pi, errors, variations = pageRank_Analysis1(www, damping)

def chart1():
    plt.figure(figsize=(8, 5))
    plt.plot(errors, linestyle='-', color='orangered')
    plt.yscale('log')  # Using log scale to better visualize error decay
    plt.title('Erreur entre le vrai vecteur propre et celui calculé par la méthode de la puissance')
    plt.xlabel('Itération')
    plt.ylabel('Erreur')
    plt.grid(True)
    plt.show()

def chart2():
    plt.figure(figsize=(8, 5))
    plt.plot(variations, linestyle='-', color='royalblue')
    plt.yscale('log')  # Using log scale to better visualize error decay
    plt.title("Variations entre le vecteur propre actuel et celui de l'itération précédente")
    plt.xlabel('Itération')
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
Temps d'exécution de PageRank (base) en fonction du nombre de noeuds, 
ce qui revient à la taille du graphe car les liens augmentent 
aussi au fûr et à mesure que le nombre de noeuds augmente.

Montre que la complexité est exponentielle par rapport au nombre de noeuds
à cause de la multiplication matricielle qui est faite en O(N^2).
"""

def chart4():
    pageRankTime = []
    num_nodes = []

    for i in range(100, 1500, 10):
        print(i)
        num_nodes.append(i)
        graph = generate_graph(i, 10)
        pageRank_start = time.time()
        pageRank(graph)
        pageRank_end = time.time()
        total_time = pageRank_end - pageRank_start
        pageRankTime.append(total_time)

    plt.figure(figsize=(8, 5))
    plt.plot(num_nodes, pageRankTime, linestyle='-', color='royalblue')
    plt.title('Complexité de PageRank en fonction du nombre de noeuds')
    plt.xlabel('Nombre de noeuds')
    plt.ylabel('Temps (sec)')
    plt.grid(True)
    plt.show()


"""
Temps d'exécution de pagerank en fonction du nombre de liens (arrêtes du graphe).

Montre que la complexité est linéaire par rapport à la densité (nombre d'arrêtes)
du graphe.
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
    plt.plot(num_edges, pageRankTime, linestyle='-', color='royalblue')
    plt.title("Complexité de PageRank en fonction du nombre d'arêtes'")
    plt.xlabel("Nombre d'arêtes")
    plt.ylabel('Temps (sec)')
    plt.grid(True)
    plt.show()


"""
Temps d'exécution de Power Method en fonction de la dimension de la matrice.ç

Explique la complexité de PageRank --> O(N^2)
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
    plt.plot(num_nodes, powerMethodTime, linestyle='-', color='royalblue')
    plt.title('Complexité de la méthode de la puissance en fonction du nombre de noeuds')
    plt.xlabel('Nombre de noeuds')
    plt.ylabel('Temps (sec)')
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
    plt.plot(num_nodes, convergence_iters, linestyle='-', color='royalblue')
    plt.title('Itération de convergence de la méthode de la puissance en fonction du nombre de noeuds')
    plt.xlabel('Nombre de noeuds')
    plt.ylim(0, 15)
    plt.ylabel('Itération')
    plt.grid(True)
    plt.show()



"""
Taux de convergence de Power Method en fonction du damping factor.

Montre que le taux de convergence est dépendant du damping factor.
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
    plt.plot(dampingFactors, convergence_iters, linestyle='-', color='royalblue')
    plt.title("Itération de convergence de la méthode de la puissance en fonction du facteur d'amortissement")
    plt.ylabel('Itération')
    plt.xlabel("Facteur d'amortissement")
    plt.grid(True)
    plt.show()



"""
Temps d'excécution de PageRank avec l'optimisation de la méthode utilisée
pour faire la multiplication matricielle en temps linéaire lorsque il y a plus 
d'éléments nuls que d'éléments non nuls dans la matrice de transition.

La complexité de PageRank devient O(k * (N + E)) mais juste pour ce type de matrices.
Pour un graphe plus dense, cette méthode est 4 fois plus lente que la méthode 
de base. Pour le web est mieux.
"""
def chart9():
    pageRankTime = []
    num_nodes = []

    for i in range(100, 2500, 10):
        print(i)
        graph = generate_graph(i, 10)
        start = time.time()
        for _ in range(1):
            pageRank_sparse(graph)
        end = time.time()
        total_time = (end - start)/1
        num_nodes.append(i)
        pageRankTime.append(total_time)

    def moving_average(data, window_size=5):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    smoothed_times = moving_average(pageRankTime, window_size=10)

    plt.figure(figsize=(8, 5))
    plt.plot(num_nodes[:len(smoothed_times)], smoothed_times, linestyle='-', color='royalblue')
    plt.title('Complexité de PageRank - Matrices creuses')
    plt.xlabel('Nombre de noeuds')
    plt.ylabel('Temps (sec)')
    plt.grid(True)
    plt.show()

## GÉNÉRATION DES GRAPHIQUES ##

if __name__ == "__main__":
    chart1()
    chart2()
    chart3()
    chart4()
    chart5()
    chart6()
    chart7()
    chart8()
    chart9()