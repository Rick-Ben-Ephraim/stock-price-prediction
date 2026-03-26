import yfinance as yf
import pandas as pd

# Download stock data
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

print(df.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title("Apple Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


from sklearn.model_selection import train_test_split
import numpy as np

# Use only Close price
data = df[['Close']]

# Create simple X (days) and y (price)
data['Prediction'] = data['Close'].shift(-10)

# Drop last rows
data.dropna(inplace=True)

X = np.array(data[['Close']])
y = np.array(data['Prediction'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained!")

predictions = model.predict(X_test)

print(predictions[:5])

import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()