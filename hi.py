from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

#histogram (part a)
df.hist(bins=15, figsize=(15, 10))
plt.suptitle('Histogram (part a)')
plt.show()

#part b
numiattri = df.shape[1] - 1  
print(f"num of numerical attributes: {numiattri}")
print("num of classes : ", len(df['target'].unique()))

#part c
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('features heatmap')
plt.show()



