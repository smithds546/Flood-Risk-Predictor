import pandas as pd

# Read the Excel file to get column names
df = pd.read_csv('DataSet.csv')

# Get the list of column names
c = df.columns.tolist()

# Read the Excel file again to get the full DataFrame
df = pd.read_csv('DataSet.csv')

# Iterate over each row and check for negative values in all columns
for x in df.index:
    for col in c:
        # Check if the value is numeric before comparing
        if pd.to_numeric(df.loc[x, col], errors='coerce') < 0:
            df = df.drop(x)
            break  # Once a negative value is found in any column, break out of the inner loop

# Convert columns to numeric, coerce errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with NaN values
df = df.dropna()

# Calculate correlation matrix
correlation_matrix = df.corr()

# Save the cleaned DataFrame to a new csv file
df.to_csv('CleanedData.csv', index=False)
correlation_matrix.to_csv('CorrelationMatrix.csv', index=False)

print(correlation_matrix)

