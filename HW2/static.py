import glob
import pandas as pd
import numpy as np

path = 'YOUR_PATH'
files = glob.glob(f'{path}/fitness-*.txt')

fitness_scores = np.array([pd.read_csv(file, header=None, sep=',')[1].min() for file in files])
mean_score = np.mean(fitness_scores)

print(mean_score)