import random
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from scipy.stats import t
from arch import arch_model
from arch.__future__ import reindexing
from preprocessing import process_from_parquet

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
def pred_algo(p, o, q, power):
    res, check_data = process_from_parquet(num_files, check_num)

    # Calculate the difference between consecutive closing values
    res['diff_close'] = res['close'].diff()
    # Drop first value which is NaN
    res = res.iloc[1:]

    # Convert timestamp to datetime format
    res['datetime'] = pd.to_datetime(res['timestamp'], unit='ms')
    # Extract minute of the day from datetime
    res['minute'] = res['datetime'].dt.minute

    # Specify the AMIRA-GARCH model
    amira_garch = arch_model(res['diff_close'], vol='GARCH', p=p, o=o, q=q, power=power, dist='t', mean='HAR', x=res[['volume', 'datetime']])
    # Fit the model
    amira_garch_fit = amira_garch.fit(disp=False)

    # Determine the length of half a day in terms of the time steps of your dataset
    horizon = int((len(res) / num_files) / 2)

    # Generate predictions for the next 'horizon' time steps
    forecast = amira_garch_fit.forecast(horizon=horizon, simulations=1000)
    predicted_values = forecast.simulations.values
    pred_mean = forecast.mean.iloc[-1]
    pred_vol = forecast.variance.iloc[-1]
    # Pass the estimated df to the t distribution
    df = (len(res['close'])-1) * (np.var(res['close']) + (np.mean(res['close'])-0)**2)/(np.var(res['close'])/(len(res['close'])-1) + (np.mean(res['close'])-0)**2)
    dist = t(df=df, loc=pred_mean, scale=np.sqrt(pred_vol))
    pred_returns = dist.rvs(size=horizon)
    pred_price = res['close'].iloc[-1] + pred_returns.cumsum()

    # Create a new x-axis that starts where the historical x-axis ends
    x_pred = res['timestamp'].iloc[-1] + np.arange(1, len(pred_price) + 1) * (res['timestamp'].iloc[-1] - res['timestamp'].iloc[-2])

    # Create data frame from the pred_price and x_pred
    data = {'pred_price': pred_price, 'timestamp': x_pred}
    pred_df = pd.DataFrame(data)
    merged_df = pd.merge_asof(pred_df, check_data[['close', 'timestamp']], on='timestamp', direction='nearest', tolerance=15000).dropna().reset_index()
    start_time = merged_df.iloc[0]['timestamp']
    end_time = merged_df.iloc[-1]['timestamp']
    time_diff = (end_time - start_time) / (3600 * 1000)

    # Truncate the dataframe such that it only has timestamps within 6 hours from the first one
    if time_diff > num_hours:
      truncate_time = start_time + num_hours * 3600 * 1000
      merged_df = merged_df.loc[merged_df['timestamp'] <= truncate_time]

    # Group truncated merged dataframe into bins according to the num_hours
    bins = pd.cut(merged_df['timestamp'], num_hours)
    grouped_df = merged_df.groupby(bins)

    # Calculate the mean and median of the residual for each bin
    merged_df['residual'] = ((merged_df['pred_price'] - merged_df['close']) / merged_df['close'] ) * 100
    grouped_df = merged_df.groupby(bins)
    agg_stats = grouped_df['residual'].agg(['mean', 'median'])

    residual_average = np.mean(np.abs(agg_stats['median'].values))

    return residual_average

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

def genetic_algo():
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
  max_generations = 1000
  mutation_rate = 0.01
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
          fitness.append(pred_algo(*individual))

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
      if p != 0 and o != 0 and pred_algo(*best_individual) < best_score:
          best_score = pred_algo(*best_individual)
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

results = []
num_runs = 150
for i in tqdm(range(num_runs)):
    best_individual, best_fitness = genetic_algo()
    results.append((best_individual, best_fitness))

individuals = [result[0] for result in results]
counter = Counter([tuple(ind) for ind in individuals])
print(dict(counter))
