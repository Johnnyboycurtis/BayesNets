I'm working on implementing the Tree Augmented Naive Bayes (TAN) algorithm as described by [Bayesian Network Classifiers](http://www.cs.technion.ac.il/~dang/journal_papers/friedman1997Bayesian.pdf) which directly builds on the [Chow-Liu algorithm](http://www.cs.cmu.edu/~guestrin/Class/15781/recitations/r10/11152007chowliu.pdf) . I have the following question: 

- Why does the algorithm use a *maximum* spanning tree vs a *minimum* spanning tree?

The algorithm as described by Friedman et al goes:

1. Compute the (conditional) [mutual information](https://en.wikipedia.org/wiki/Mutual_information#Definition), $I(A_i, A_j | C)$,  between each pair of attributes (predictors)

2. Build a complete undirected graph in which the verticies are the attributes $A_1, A_2, ..., A_n$. Annotate the weight of an edge connecting $A_i \to A_j$ by $I(A_i, A_j | C)$.

3. Build a *maximum* weighted spanning tree

4. Transform the resulting undirected tree to a directed one by choosing a root variable and setting the direction of all edges to be outward from it.

5. Contstruct a TAN model by adding a vertex labeled by $C$ and adding an arc from $C$ to each $A_i$.

The reason for using a TAN model vs. ordinary Naive Bayes model is that the the ordinary Naive Bayes model assumes each attribute is independent of each other, $A_i \perp A_j$. This is of course not realistic. The TAN model tries to overcome this problem by 
> [augmenting] the naive Bayes structure with edges among the attributes, when needed, thus dispensing its strong assumptions about independence.

> In an augmented struct, an edge from $A_i \to A_j$ implies that the influence of $A_i$ on the assessment of the class variable also depends on the value of $A_j$


The mutual information measure, $I(A_i, A_j | C) \geq 0$, basically measures how similar two distributions may be. The closer to zero the greater similarity. Given that $I(A_i, A_j | C)$ will act as weights, I'd imagine you would want to use a *minimum* spanning tree, so why does the algorithm specify building a *maximum* spanning tree



AN UPDATE:
I think the reason you want to use a maximum spanning tree is because you want to compute the product of *almost* independent attributes...
If you used the minimum spanning tree, you'd sure be multiplying the most dependent variables, choosing the worst path?
Let's ask this later



