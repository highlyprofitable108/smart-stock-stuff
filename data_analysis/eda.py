from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# MongoDB connection details
mongo_uri = 'mongodb://localhost:27017/'  # Update this to your MongoDB connection string
db_name = 'stock_data'
collection_name = 'model_data'

# Connect to the MongoDB client
client = MongoClient(mongo_uri)

# Select the database and collection
db = client[db_name]
collection = db[collection_name]

# Fetch the data from MongoDB and load it into a DataFrame
data = list(collection.find())
df = pd.json_normalize(data)

# Drop the sentiment_score column, and any other columns not needed
df = df.drop(columns=['sentiment_score'], errors='ignore')

# Convert 'date' to datetime format for easier manipulation
df['date'] = pd.to_datetime(df['date'])

# Initial EDA
# Summary statistics for numeric columns
print(df.describe())

# Distribution of adjusted close prices
plt.figure(figsize=(10, 6))
sns.histplot(df['adjusted_close'], bins=30, kde=True)
plt.title('Distribution of Adjusted Close Prices')
plt.xlabel('Adjusted Close Price')
plt.ylabel('Frequency')
plt.show()

# Time-series plot of adjusted close prices
plt.figure(figsize=(14, 7))
sns.lineplot(x='date', y='adjusted_close', data=df, marker='o')
plt.title('Time Series of Adjusted Close Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.xticks(rotation=45)
plt.show()

# Correlation matrix of numeric features
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# Remember to close the MongoDB connection when done
client.close()
