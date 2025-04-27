import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("clean_stolen_vehicless.csv")
df.dropna(inplace=True)
df['date_stolen'] = pd.to_datetime(df['date_stolen'], errors='coerce')

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

plt.figure()
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.tight_layout()

plt.figure()
sns.countplot(data=df, x='vehicle_type', order=df['vehicle_type'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Vehicle Type Distribution')
plt.tight_layout()

plt.figure()
sns.countplot(data=df, x='color', order=df['color'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Vehicle Color Distribution')
plt.tight_layout()

plt.figure()
sns.histplot(df['model_year'], bins=30, kde=True)
plt.title('Distribution of Model Years')
plt.tight_layout()

df['year_month'] = df['date_stolen'].dt.to_period('M')
monthly_counts = df['year_month'].value_counts().sort_index()

plt.figure()
monthly_counts.plot(kind='line', marker='o')
plt.title('Number of Vehicles Stolen Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

plt.figure()
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()

plt.show()
