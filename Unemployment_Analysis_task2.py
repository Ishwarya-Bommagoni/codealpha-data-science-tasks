import pandas as pd
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("unemployment.csv")

# clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# show basic data
print("First 5 rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

# simple graph
df['Estimated Unemployment Rate (%)'].plot()

plt.title("Unemployment Rate Over Time")
plt.xlabel("Index")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# better graph (average by date)
df.groupby('Date')['Estimated Unemployment Rate (%)'].mean().plot()

plt.title("Average Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.show() 
