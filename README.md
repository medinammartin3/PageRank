# PageRank

Implementation and analysis of the PageRank algorithm in Python. This repository includes  both a standard implementation and a sparse-matrix optimized version.



## Background & Algorithm 

PageRank is a link-analysis algorithm originally used by Google to rank web pages by importance based on the structure of links between them. 
Pages that are linked to by many “important” pages receive a high rank score themselves.

The score distribution represents a stationary distribution of a Markov chain where transitions follow outgoing links with a damping factor.



## Project Structure 

```text
├── pagerank.py         # Core PageRank implementation (dense matrix) 
├── pagerank_sparse.py  # Sparse matrix PageRank for large graphs 
├── charts.py           # Plotting utilities for results and convergence
├── paper.pdf           # Technical analysis of PageRank's computational complexity
├── README.md           # Project overview and instructions 
├── .gitignore          # Files and folders to ignore in Git 
└── requirements.txt    # Python dependencies 
 ``` 



## Features

> [!NOTE]
> $n$ is the number of nodes (pages) and $E$ is the number of edges (links) in the graph.

- Standard PageRank using dense adjacency structures. Complexity : $\mathcal{O}(n^2 + E)$.
- Sparse PageRank for efficiency on large graphs. Complexity : $\mathcal{O}(n + E)$
- Visualization tools in `charts.py` for plotting rankings and convergence
- Support for damping factor customization



## Installation & Setup 
1. **Clone the repository** 

```bash 
git clone https://github.com/medinammartin3/PageRank.git 
cd PageRank 
``` 

2. **Install dependencies** 
```bash 
pip install -r requirements.txt 
```



## Usage Examples 

TODO


