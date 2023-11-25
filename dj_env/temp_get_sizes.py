import numpy as np

# input file path to whatever data is being trained on 
data = np.load("user_taste/data/completed_score_matrix.npy")

# Find the data size
rows, columns = data.shape
print("rows (users):", rows) # rows are users
print("columns (songs):", columns) # columns are songs

# Find the maximum and minimum values
max_value = np.max(data)
min_value = np.min(data)

print("Maximum Value:", max_value)
print("Minimum Value:", min_value)
