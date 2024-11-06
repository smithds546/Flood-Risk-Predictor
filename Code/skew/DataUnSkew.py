import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the dataset
df = pd.read_csv('cleanedData.csv')

# Extract the 'LDP' column
column = df['Index flood']

# Apply logarithmic transformation to unskew the data
logged = np.log(column)


# Plot the distribution of 'AREA' after logarithmic transformation
sns.histplot(logged, kde=True)
plt.title('Distribution of Index flood (After Logarithmic Transformation)')
plt.xlabel('Log(Index flood)')
plt.ylabel('Frequency')
plt.show()

##df = np.log(df)
##print(df.skew(axis = 0))

