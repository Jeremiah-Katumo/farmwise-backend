import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump
from keras.models import Sequential
from keras.layers import Dense
import joblib

# Sample data
data = pd.DataFrame({
    'soil_moisture': [500, 600, 450, 700],
    'temperature': [23, 25, 21, 30],
    'humidity': [60, 55, 70, 50],
    'yield': [2.5, 2.8, 2.0, 3.0]
})

X = data[['soil_moisture', 'temperature', 'humidity']]
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict with new data
new_data = pd.DataFrame({'soil_moisture': [550], 'temperature': [24], 'humidity': [65]})
predicted_yield = model.predict(new_data)
print("Predicted yield:", predicted_yield[0])
# Save the random forest model
joblib.dump(model, 'random_forest_model.pkl')


# train a deep learning model

deep_model = Sequential()
deep_model.add(Dense(64, input_dim=3, activation='relu'))
deep_model.add(Dense(32, activation='relu'))
deep_model.add(Dense(1, activation='linear'))
deep_model.compile(optimizer='adam', loss='mean_squared_error')
deep_model.fit(X_train, y_train, epochs=50, batch_size=5)
deep_model.save("model.pkl")

# Save the keras model for later use
dump(deep_model, 'deep_model.pkl')