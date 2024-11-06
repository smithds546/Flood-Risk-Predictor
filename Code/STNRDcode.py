import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data from cleanedData.csv
data = pd.read_csv('cleanedData.csv')

# Extract features (X) and target (y)
X = data.drop(columns=['Index flood'])  # Assuming 'Index flood' is the target column
y = data['Index flood']

# Combine features and target for scaling
combined_data = pd.concat([X, y], axis=1)

# Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_data)

# Create a DataFrame with scaled data
scaled_df = pd.DataFrame(scaled_data, columns=combined_data.columns)

# Write the scaled data to a new file called EqualData.csv
scaled_df.to_csv('EqualData.csv', index=False)
