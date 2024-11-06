import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('cleanedData.csv')

# Scatter plot to compare correlation between 'AREA' and 'LDP'
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='RMED-1D', y='SAAR')
plt.title('Correlation between RMED-1D and SAAR')
plt.xlabel('RMED-1D')
plt.ylabel('SAAR')
plt.show()
