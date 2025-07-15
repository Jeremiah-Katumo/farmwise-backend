import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os, datetime
from dotenv import load_dotenv


def build_train_pest_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['mae', 'accuracy'])
    
    # logdir = f"logs/pest/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])
    
    return model.save("pest_dl_model.h5")

def predict_pest_dl(data):
    model = load_model("pest_dl_model.h5")
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return {"predicted_pest": int(np.round(prediction[0][0]))}


def build_and_train_yield_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])
    
    return model.save("yield_dl_model.h5")

def predict_yield_dl(data):
    model = load_model("yield_dl_model.h5")
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return {"predicted_yield": float(prediction[0][0])}