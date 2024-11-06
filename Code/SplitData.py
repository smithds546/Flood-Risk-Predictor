import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('EqualData.csv')

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate the sizes for each set
total_size = len(df)
train_size = int(0.6 * total_size)
validation_size = int(0.5 * (total_size - train_size))

# Split the dataset
train_data = df.iloc[:train_size]
validation_data = df.iloc[train_size:train_size + validation_size]
test_data = df.iloc[train_size + validation_size:]

# Save the datasets to CSV files
train_data.to_csv('TrainData.csv', index=False)
validation_data.to_csv('ValidationData.csv', index=False)
test_data.to_csv('TestData.csv', index=False)
