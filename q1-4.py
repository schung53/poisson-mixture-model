import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import factorial

df = pd.read_csv('./pmm_q1.tsv')
data = df.to_numpy()
data_arr = [x[0] for x in data]

## Use K-means clustering to initialize model parameters
kmeans = KMeans(n_clusters=3, n_init='auto').fit(data.reshape(-1,1))
kmeans_result = kmeans.predict(data.reshape(-1,1))

cluster_0 = []
cluster_1 = []
cluster_2 = []

## For each cluster, group and determine lambdas and pis
for idx, val in enumerate(kmeans_result):
  if val == 0:
    cluster_0.append(data[idx][0])
  elif val == 1:
    cluster_1.append(data[idx][0])
  else:
    cluster_2.append(data[idx][0])

def poissonMLE(cluster):
  return np.mean(cluster)

def calculate_pi(cluster):
  return len(cluster) / len(data)

## Initial parameters setup
lambdas = [poissonMLE(cluster_0), poissonMLE(cluster_1), poissonMLE(cluster_2)]
pis = [calculate_pi(cluster_0), calculate_pi(cluster_1), calculate_pi(cluster_2)]
alphas = [2, 2, 2]

n_iter = 2000

## Use the multinomial model to pick a group based on weights
def randomize_z(pi_arr):
  groups = np.random.multinomial(1, pi_arr)
  for idx, val in enumerate(groups):
    if val == 1:
      return idx

x = random.choice(data_arr)
z = randomize_z(pis)

## Sample elements are tuples of form: (x, z, [lambdas], [pis])
samples = []
samples.append((x, z, lambdas, pis))

for i in range(n_iter):
  ## The curr lambda to use is dictated by the curr z
  curr_lambda = lambdas[z]
  ## Use the curr lambda to randomize poisson
  new_x = np.random.poisson(curr_lambda, size=1)[0]
  ## For the k=3 groups, generate lambdas using gamma dist.
  new_lambdas = []
  for i in range(3):
    new_lambda  = np.random.gamma(2, 0.5, size=1)[0]
    new_lambdas.append(new_lambda)
  ## Generate new pi values using dirichlet dist.
  new_pis = np.random.dirichlet(alphas, size=1)[0]
  ## Generate new z with the new pi values using the categorical dist.
  new_z = randomize_z(new_pis)

  ## Log the newly generated values and replace curr vals with them
  samples.append((new_x, new_z, new_lambdas, new_pis))
  x, z, lambdas, pis = new_x, new_z, new_lambdas, new_pis

print(samples)

