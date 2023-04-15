import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Get the stock data for TCS
import yfinance as yf
tickerSymbol = 'TCS.NS'
start_date = '2000-04-12'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
data = yf.download(tickerSymbol, start=start_date, end=end_date)

# Create a new DataFrame with only the 'Close' column and apply log function to convert to log-normal distribution
data = pd.DataFrame({'Close': np.log(data['Close'])})

# Standardize the log-normal data
scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(dataset_scaled) * 0.8)
test_size = len(dataset_scaled) - train_size
train_data, test_data = dataset_scaled[0:train_size,:], dataset_scaled[train_size:len(dataset_scaled),:]

# Define the function to create the dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i+look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Create the training and testing datasets
look_back = 60
train_X, train_Y = create_dataset(train_data, look_back)
test_X, test_Y = create_dataset(test_data, look_back)

# Reshape the input data for LSTM
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(look_back, 1)))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the LSTM model
history = model.fit(train_X, train_Y, epochs=100, batch_size=64, validation_split=0.3, verbose=2)

# Make predictions using the LSTM model
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# Invert the scaling of the predictions
train_predict = scaler.inverse_transform(train_predict)
train_Y = scaler.inverse_transform([train_Y])
test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])

# Plot the actual closing price vs the predicted closing price
train_predict_plot = np.empty_like(dataset_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict

test_predict_plot = np.empty_like(dataset_scaled)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(look_back*2):len(dataset_scaled), :] = test_predict

plt.figure(figsize=(16,8))
plt.plot(scaler.inverse_transform(dataset_scaled), label='Actual')
plt.plot(train_predict_plot, label='Training Prediction')
plt.plot(test_predict_plot, label='Testing Prediction')
plt.legend()
plt.show()

# Invert the log-normal predictions to normal predictions
normal_train_predict = np.exp(train_predict)
normal_train_Y = np.exp(train_Y)
normal_test_predict = np.exp(test_predict)
normal_test_Y = np.exp(test_Y)

# Plot the actual closing price vs the predicted closing price in normal scale
normal_train_predict_plot = np.empty_like(dataset_scaled)
normal_train_predict_plot[:, :] = np.nan
normal_train_predict_plot[look_back:len(normal_train_predict)+look_back, :] = normal_train_predict

normal_test_predict_plot = np.empty_like(dataset_scaled)
normal_test_predict_plot[:, :] = np.nan
normal_test_predict_plot[len(normal_train_predict)+(look_back*2):len(dataset_scaled), :] = normal_test_predict

plt.figure(figsize=(16,8))
plt.plot(np.exp(scaler.inverse_transform(dataset_scaled)), label='Actual (Normal Scale)')
plt.plot(normal_train_predict_plot, label='Training Prediction (Normal Scale)')
plt.plot(normal_test_predict_plot, label='Testing Prediction (Normal Scale)')
plt.legend()
plt.show()

