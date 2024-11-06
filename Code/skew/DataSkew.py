import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('cleanedData.csv')

# Extract the 'AREA' column
column = df['AREA']

# Calculate skewness for the 'AREA' column
skewness = column.skew()

print(skewness)

# Plot the distribution of 'AREA' using a distplot
sns.displot(df['AREA'])
plt.title('Distribution of AREA')
plt.xlabel('AREA')
plt.ylabel('Frequency')
plt.show()


