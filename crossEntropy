import math
import numpy as np
x_input=np.array([(0.1,0.5,0.2),
                (0.2,0.3,0.1),
                (0.7,0.4,0.2),
                (0.1,0.4,0.3)])
w_weight=np.array([0.4,0.2,0.6])
y_input=np.array([1,0,1,0])
arr=np.array([(0,0),(0,0),(0,0),(0,0)])
matrix=[]
threshold=0.5

def step(weighted_sum):
  if weighted_sum > threshold:
    return 1
  else:
    return 0

def perceptron():
  for i in range(4):
    row=[]
    weighted_sum=0
    weighted_sum += np.dot(x_input[i],w_weight)
    row.append(weighted_sum)
    row.append(y_input[i])
    matrix.append(row)
  return matrix
  
def cross_entropy(input_data):
  loss=0
  n=len(input_data)
  for entry in input_data:
    w_sum=entry[0]
    y=entry[1]
    loss += -((y*math.log10(w_sum))+((1-y)*math.log10(1-w_sum)))/n
  return loss

output=perceptron()
print(output)
error_term=cross_entropy(output)
print(error_term)
