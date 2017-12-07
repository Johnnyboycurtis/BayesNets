import sys
sys.path.append("../src/")


if __name__ == "__main__":
    from Probs2 import *
    import pandas as pd
    import numpy as np

    x = np.random.rand(100)
    s = pd.Series(x)

    p = Probs(s, nonparemetric = False)
    pval = p.Evaluate(0.002)
    print(pval, type(pval))
