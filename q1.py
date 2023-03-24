import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import poisson, gamma, dirichlet
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
initial_lambdas = [poissonMLE(cluster_0), poissonMLE(cluster_1), poissonMLE(cluster_2)]
initial_pis = [calculate_pi(cluster_0), calculate_pi(cluster_1), calculate_pi(cluster_2)]
alphas = [2, 2, 2]

## Termination criterion values
## log-likelihoods + summation of log-priors
log_likelihoods = []

## Keep a log of all lambdas and pis generated so far
lambda_history = []
lambda_history.append(initial_lambdas)
pi_history = []
pi_history.append(initial_pis)

def calculate_log_ll(lambdas, pis):
  log_ll = 0
  for i in range(len(data)):
    x_i = data[i][0]
    ll_sum = 0
    for k in range(3):
      lambda_k = lambdas[k]
      pi_k = pis[k]
      ll_sum += pi_k * poisson.pmf(x_i, lambda_k)
    log_ll += math.log(ll_sum)
  for k in range(3):
    lambda_k = lambdas[k]
    log_ll += math.log(gamma.pdf(lambda_k, 2, loc=0, scale=0.5))
  log_ll += math.log(dirichlet.pdf(pi_history[-1], alphas))
  log_likelihoods.append(log_ll)

## Calculate the log-likelihood + log-prior based on initial params
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
  ## Calculate the new log-likelihood + log-prior
  calculate_log_ll(new_lambda_vector, new_pi_vector)

delta = math.inf
epsilon = 1e-5

## Iterate until termination condition is met
while delta > epsilon:
  executeEM()
  delta = abs(log_likelihoods[-1] - log_likelihoods[-2])

print("\nInitial lambdas: " + str(initial_lambdas))
print("Final lambdas: " + str(lambda_history[-1]))
print("\nInitial pis: " + str(initial_pis))
print("Final pis: " + str(pi_history[-1]))

fig, (ax1, ax2) = plt.subplots(1, 2)
x = np.arange(0, 26, 0.1)

for k in range(3):
  lambda_k = lambda_history[-1][k]
  pmf = np.exp(-lambda_k) * np.power(lambda_k, x) / factorial(x)
  pmf = np.round(pmf, 5)
  ax1.plot(x, pmf)

ax1b = ax1.twinx()
ax1b.hist(data_arr, 25, edgecolor = 'black', alpha=0.5)

ax1.set_xlabel('Data Value')
ax1.set_ylim([0, 1.5])
ax1.set_ylabel('Density')
ax1b.set_ylabel('Frequency')

x_ll = np.arange(0, len(log_likelihoods))
ax2.plot(x_ll, log_likelihoods)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Log Likelihood + Log Prior')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax1.set_title('Histogram of Raw Data and Final Mixture Components')
ax2.set_title('Log Likelihood + Log Prior Over All Iterations')
plt.show()
