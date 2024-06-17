import numpy as np #for the math part
import pandas as pd #to organize, visualize

#Create random features, targets, weights
rg=np.random.default_rng()
print(rg)

def generate_data(n_features, n_values):
  features=rg.random((n_features, n_values))
  weights=rg.random((1,n_values))[0]
  targets=np.random.choice([0,1],n_features)
  data=pd.DataFrame(features,columns=["x0","x1","x2"])
  data["targets"]=targets
  return data, weights
