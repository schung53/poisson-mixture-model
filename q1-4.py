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

## Use the multinomial, since categorical is a special case of multinomial
def randomize_z(pi_arr):
  groups = np.random.multinomial(1, pi_arr)
  for idx, val in enumerate(groups):
    if val == 1:
      return idx

x = random.choice(data_arr)
z = randomize_z(pis)

## Save iteration results in arrays
x_history = [x]
z_history = [z]
lambda_history = [lambdas]
pi_history = [pis]
lambda_avgs = [lambdas]
pi_avgs = [pis]

def avg_lambdas(new, n):
  curr = lambda_avgs[-1]
  new_avgs = [
    curr[0]*(n-1)/n + new[0]/n,
    curr[1]*(n-1)/n + new[1]/n,
    curr[2]*(n-1)/n + new[2]/n
  ]
  lambda_avgs.append(new_avgs)

def avg_pis(new, n):
  curr = pi_avgs[-1]
  new_avgs = [
    curr[0]*(n-1)/n + new[0]/n,
    curr[1]*(n-1)/n + new[1]/n,
    curr[2]*(n-1)/n + new[2]/n
  ]
  pi_avgs.append(new_avgs)

for i in range(n_iter):
  ## The old lambda to use is dictated by the the old z
  old_lambda = lambda_history[-1][z_history[-1]]
  ## Use the old lambda to randomize poisson
  new_x = np.random.poisson(old_lambda, size=1)[0]

  ## For the k=3 groups, generate new lambdas
  old_lambdas = lambda_history[-1]
  new_lambdas = []
  for k in range(3):
    new_lambda = np.random.gamma(2, 0.5, size=1)[0] * np.random.poisson(old_lambdas[k])
    new_lambdas.append(new_lambda)

  old_pis = pi_history[-1]
  ## Generate new z
  new_z = randomize_z(old_pis[0:-1])

  ## Generate new pi values
  triplet = np.random.dirichlet(alphas, size=1)[0]
  new_pis = [d * new_z for d in triplet]

  ## Record the newly generated values and replace old vals with them
  x_history.append(new_x)
  z_history.append(new_z)
  lambda_history.append(new_lambdas)
  pi_history.append(new_pis)
  avg_lambdas(new_lambdas, i + 2)
  avg_pis(new_pis, i + 2)

x_axis = np.arange(n_iter)

lambda_0_history = []
lambda_1_history = []
lambda_2_history = []

pi_0_history = []
pi_1_history = []
pi_2_history = []

lambda_0_avgs = []
lambda_1_avgs = []
lambda_2_avgs = []

pi_0_avgs = []
pi_1_avgs = []
pi_2_avgs = []

for i in range(n_iter):
  lambda_0_history.append(lambda_history[i][0])
  lambda_1_history.append(lambda_history[i][1])
  lambda_2_history.append(lambda_history[i][2])
  pi_0_history.append(pi_history[i][0])
  pi_1_history.append(pi_history[i][1])
  pi_2_history.append(pi_history[i][2])
  lambda_0_avgs.append(lambda_avgs[i][0])
  lambda_1_avgs.append(lambda_avgs[i][1])
  lambda_2_avgs.append(lambda_avgs[i][2])
  pi_0_avgs.append(pi_avgs[i][0])
  pi_1_avgs.append(pi_avgs[i][1])
  pi_2_avgs.append(pi_avgs[i][2])

fig, axs = plt.subplots(3)
axs[0].set_title("Lambda_k Values Over Iterations")
axs[0].plot(x_axis, lambda_0_history, '.', alpha=0.5)
axs[1].plot(x_axis, lambda_1_history, '.', alpha=0.5)
axs[2].plot(x_axis, lambda_2_history, '.', alpha=0.5)

fig2, axs2 = plt.subplots(3)
axs2[0].set_title("Pi_k Values Over Iterations")
axs2[0].plot(x_axis, pi_0_history, '.', alpha=0.5)
axs2[1].plot(x_axis, pi_1_history, '.', alpha=0.5)
axs2[2].plot(x_axis, pi_2_history, '.', alpha=0.5)

fig3, axs3 = plt.subplots(3)
axs3[0].set_title("Lambda_k Running Average")
axs3[0].plot(x_axis, lambda_0_avgs, alpha=0.5)
axs3[1].plot(x_axis, lambda_1_avgs, alpha=0.5)
axs3[2].plot(x_axis, lambda_2_avgs, alpha=0.5)
axs3[0].set_ylim([0, 1.5])
axs3[1].set_ylim([0, 10])
axs3[2].set_ylim([0, 15])

fig4, axs4 = plt.subplots(3)
axs4[0].set_title("Pi_k Running Average")
axs4[0].plot(x_axis, pi_0_avgs, alpha=0.5)
axs4[1].plot(x_axis, pi_1_avgs, alpha=0.5)
axs4[2].plot(x_axis, pi_2_avgs, alpha=0.5)
axs4[0].set_ylim([0, 0.5])
axs4[1].set_ylim([0, 0.5])
axs4[2].set_ylim([0, 0.5])

plt.show()
