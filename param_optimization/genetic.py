import sys, os
sys.path.append(os.getcwd())

import math
import random
import pandas as pd
import numpy as np
import helpers.analysis as analysis
from collections import Counter
from tqdm import tqdm
from scipy.stats import t
from arch import arch_model
from arch.__future__ import reindexing
from helpers.preprocessing import process_from_parquet

# Global Variables
num_files = 21
check_num = 1
num_hours = 6
# Model Variables
p_symmetric_lag = [0,1,2,3]
o_asymmetric_lag = [0,1,2,3]
q_volatility_lag = [0,1,2,3]
# power_lag = [0,1,2]
power_lag = [2]

# Define the prediction algorithm
def pred_algo(p, o, q, power, choice):
    res, check_data = process_from_parquet(num_files, check_num)
    res = res.assign(diff_open=lambda x: x['open'].diff(),
                    diff_high=lambda x: x['high'].diff(),
                    diff_low=lambda x: x['low'].diff(),
                    diff_close=lambda x: x['close'].diff(),
                    datetime=lambda x: pd.to_datetime(x['timestamp'], unit='ms')
                    )[1:]

    # Specify the GJR-GARCH model and fit it
    exogenous = res[['volume', 'datetime', 'diff_open', 'diff_high', 'diff_low']]
    # exogenous = res[['volume', 'datetime']]
    gjr_garch_fit = arch_model(res['diff_close'], vol='GARCH', p=p, o=o, q=q, power=2, dist='t', mean='HAR', x=exogenous).fit(disp=False)

    # Determine the length of half a day in terms of the time steps of your dataset
    horizon = int((len(res) / num_files) / num_hours)

    # Generate predictions for the next 'horizon' time steps
    forecast = gjr_garch_fit.forecast(horizon=horizon, simulations=1000)
    # Calculate degrees of freedom for t distribution
    df = (len(res['close'])-1) * (np.var(res['close']) + (np.mean(res['close'])-0)**2)/(np.var(res['close'])/(len(res['close'])-1) + (np.mean(res['close'])-0)**2)
    # Until merged_df returns data successfully generate price predictions
    while True:
        # Generate price predictions using t distribution
        pred_returns = t(df=df, loc=forecast.mean.iloc[-1], scale=np.sqrt(forecast.variance.iloc[-1])).rvs(size=horizon)
        pred_price = res['close'].iloc[-1] + pred_returns.cumsum()

        # Create a new x-axis that starts where the historical x-axis ends
        x_pred = res['timestamp'].iloc[-1] + np.arange(1, len(pred_price) + 1) * (res['timestamp'].iloc[-1] - res['timestamp'].iloc[-2])

        # Create data frame from the pred_price and x_pred
        pred_df = pd.DataFrame({'pred_price': pred_price, 'timestamp': x_pred})
        merged_df = pd.merge_asof(pred_df, check_data[['close', 'timestamp']], on='timestamp', direction='nearest', tolerance=15000).dropna().reset_index()
        if(len(merged_df) > 0): break

    # Truncate the dataframe such that it only has timestamps within 'num_hours' hours from the first one
    time_diff = (merged_df.iloc[-1]['timestamp'] - merged_df.iloc[0]['timestamp']) / (3600 * 1000)
    if time_diff > num_hours:
        truncate_time = merged_df.iloc[0]['timestamp'] + num_hours * 3600 * 1000
        merged_df = merged_df.loc[merged_df['timestamp'] <= truncate_time]

    # Return directional accuracy or RSME based on user choice
    if choice == 0:
        return analysis.directional_accuracy(merged_df['close'], merged_df['pred_price'])
    else:
        return analysis.rmse(merged_df)

def crossover(parent1, parent2):
    # Choose a random index for the crossover point
    crossover_point = np.random.randint(0, len(parent1))
    # Combine the genes of the two parents
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutation(child):
  # Choose a random index to mutate
  mutation_point = np.random.randint(0, len(child))
  # Mutate the gene at the chosen index
  if mutation_point == 0:
      child[mutation_point] = random.choice(p_symmetric_lag)
  elif mutation_point == 1:
      child[mutation_point] = random.choice(o_asymmetric_lag)
  elif mutation_point == 2:
      child[mutation_point] = random.choice(q_volatility_lag)

  return child

def genetic_algo(opt_type):
  # Initialize the population
  # Max population is 4^3 for the four options for the three parameters
  # -4 options for the fact that both p and o cannot both be 0 
  population_size = 64-4
  population = []
  for i in range(population_size):
    p = random.choice(p_symmetric_lag)
    o = random.choice(o_asymmetric_lag)
    while p == 0 and o == 0:
      p = random.choice(p_symmetric_lag)
      o = random.choice(o_asymmetric_lag)
    q = random.choice(q_volatility_lag)
    # power = random.choice(power_lag)
    population.append([p, o, q, 2])

  # Set the parameters for the GA
  max_generations = 500
  mutation_rate = 0.02
  convergence_threshold = 0.001

  # Run the GA
  generation = 0
  best_score = float('inf')
  while generation < max_generations:
      # Evaluate fitness
      fitness = []
      for individual in population:
          p, o, q, power = individual
          # For model to run correctly p and o cannot both be 0
          if p <= 0 and o <= 0:
              continue
          fitness.append(pred_algo(*individual, opt_type))

      # Eliminate any NaN values
      fitness = np.array(fitness)
      fitness = fitness[~np.isnan(fitness)]

      # Scale fitness values to prevent premature convergence
      fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness))

      # Select the best individuals
      fitness = fitness.tolist()
      population = [x for _,x in sorted(zip(fitness, population))]
      best_individual = population[0]

      # Check for stop criteria
      p, o, q, power = best_individual
      # For model to run correctly p and o cannot both be 0
      # > if want higher values, < for lower
      if p != 0 and o != 0 and pred_algo(*best_individual, opt_type) > best_score:
          best_score = pred_algo(*best_individual, opt_type)
          best_individual = population[0]
      else:
          break

      # Crossover and mutation
      new_population = []
      for i in range(population_size):
          # Select parents using fitness proportional selection
          parent1 = random.choices(population, weights=fitness, k=1)[0]
          parent2 = random.choices(population, weights=fitness, k=1)[0]
          
          child = crossover(parent1, parent2)
          if random.random() < mutation_rate:
              child = mutation(child)
          
          new_population.append(child)

      # Replace the old population with the new one
      population = new_population

      # Increment the generation counter
      generation += 1
    
  # Return best_individual and best_score
  return best_individual, best_score

print("What sort of optimization?")
opt_type = int(input("0: Directional Accuracy - 1: Root Mean Square Error"))
results = []
num_runs = 250
for i in tqdm(range(num_runs)):
    best_individual, best_fitness = genetic_algo(opt_type)
    results.append((best_individual, best_fitness))

individuals = [result[0] for result in results]
counter = Counter([tuple(ind) for ind in individuals])
print(dict(counter))
