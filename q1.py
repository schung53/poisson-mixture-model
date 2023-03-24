import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import poisson
from scipy.special import factorial

df = pd.read_csv('./pmm_q1.tsv')
data = df.to_numpy()
data_arr = [x[0] for x in data]

kmeans = KMeans(n_clusters=3, n_init='auto').fit(data.reshape(-1,1))
kmeans_result = kmeans.predict(data.reshape(-1,1))

cluster_0 = []
cluster_1 = []
cluster_2 = []

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
initial_lambdas = [poissonMLE(cluster_0), poissonMLE(cluster_1), poissonMLE(cluster_2)]
initial_pis = [calculate_pi(cluster_0), calculate_pi(cluster_1), calculate_pi(cluster_2)]
log_likelihoods = []

lambda_history = []
lambda_history.append(initial_lambdas)
pi_history = []
pi_history.append(initial_pis)

def calculate_log_ll(lambdas, pis):
  log_ll = 0
  for i in range(len(data)):
    x_i = data[i][0]
    sub_sum = 0
    for k in range(3):
      lambda_k = lambdas[k]
      pi_k = pis[k]
      sub_sum += pi_k * poisson.pmf(x_i, lambda_k)
    log_ll += math.log(sub_sum)
  log_likelihoods.append(log_ll)

calculate_log_ll(initial_lambdas, initial_pis)

def calculate_r_ik(i, k):
  x_i = data[i][0]
  numerator = 0
  denominator = 0
  lambdas = lambda_history[-1]
  pis = pi_history[-1]
  for idx in range(3):
    pi_k = pis[idx]
    lambda_k = lambdas[idx]
    contribution = pi_k * poisson.pmf(x_i, lambda_k)
    denominator += contribution
    if idx == k:
      numerator = contribution
  return (numerator / denominator)

def executeEM():
  r_iks = [[], [], []]    ## k groups of i elements

  ## E-step: compute the posterior probabilities
  for i in range(len(data)):
    r_iks[0].append(calculate_r_ik(i, 0))
    r_iks[1].append(calculate_r_ik(i, 1))
    r_iks[2].append(calculate_r_ik(i, 2))

  new_lambda_vector = []
  new_pi_vector = []

  def update_parameters(k):
    numerator = 0
    N_k = 0
    for i in range(len(data)):
      r_ik = r_iks[k][i]
      x_i = data[i][0]
      numerator += r_ik * x_i
      N_k += r_ik
    new_lambda_vector.append(numerator / N_k)
    new_pi_vector.append(N_k / len(data))

  ## M-step
  for k in range(3):
    update_parameters(k)

  lambda_history.append(new_lambda_vector)
  pi_history.append(new_pi_vector)
  calculate_log_ll(new_lambda_vector, new_pi_vector)

delta = math.inf
epsilon = 1e-5

while delta > epsilon:
  executeEM()
  delta = abs(log_likelihoods[-1] - log_likelihoods[-2])

print("Initial lambdas: " + str(initial_lambdas))
print("Final lambdas: " + str(lambda_history[-1]))
print("\n")
print("Initial pis: " + str(initial_pis))
print("Final pis: " + str(pi_history[-1]))

fig, ax1 = plt.subplots()
x = np.arange(0, 26, 0.1)
for k in range(3):
  lambda_k = lambda_history[-1][k]
  pmf = np.exp(-lambda_k) * np.power(lambda_k, x) / factorial(x)
  pmf = np.round(pmf, 5)
  ax1.plot(x, pmf)

ax2 = ax1.twinx()
n, bins, patches = ax2.hist(data_arr, 25, edgecolor = 'black', color='blue', alpha=0.5)
ax1.set_xlabel('Data Value')
ax1.set_ylim([0, 1.5])
ax1.set_ylabel('Density')
ax2.set_ylabel('Frequency')
plt.title('Histogram of Raw Data and Final Mixture Components')
plt.show()
