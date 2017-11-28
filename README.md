# BayesNets


`BayesNets` contains various implementations of the Naive Bayes through the perspective of Naive Bayes as a special version of a Bayesian Network.

## Algorithms

1. Classic Naive Bayes with Nonparametric Density Estimation

2. Selecive Naive Bayes algorithm (k-fold CV)

3. Tree Augmented Naive Bayes (treenb: discrete data and kdetree: continuous data)

4. Hierarchical Naive Bayes (in development)


## Naive Bayes Usage
    
	import numpy as np
	import pandas as pd
    import NaiveBayesNets as nbn

	df = pd.read_csv("data/Pima.tr.csv")
	class_col_name = 'type'
	nbmodel = nbn.NaiveBayes(df, class_col_name)

	preds = nbmodel.Predict(df) ## prediction probs
	preds[class_col_name] = preds.idxmax(axis = 1) ## to get class predictions
	preds.head()

	accuracy = np.mean(preds[class_col_name].values == preds[class_col_name])
	print(accuracy)




## Dependencies

1. Numpy

2. Scipy

3. Pandas

4. Itertools

5. Matplotlib

6. Networkx


## Reference:

See [TAN](http://ai.stanford.edu/~moises/tutorial/sld164.htm) for an example of a Tree Augmented Naive Bayes.

<div align="center">
  <img src="http://ai.stanford.edu/~moises/tutorial/img164.GIF"><br>
</div>

