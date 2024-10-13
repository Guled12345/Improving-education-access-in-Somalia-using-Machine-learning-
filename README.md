# Improving Education Access in Somalia Using Machine Learning

## Project Overview
This project aims to create a predictive machine learning model that identifies students in Somalia at risk of dropping out or receiving insufficient education due to socioeconomic or geographical factors.

## Project Setup
Before proceeding, ensure you have the necessary dependencies installed. You can install the required libraries using:

## bash
```pip install tensorflow scikit-learn```
I also recommend using Google Colab for this project to leverage GPU support.

## Dataset Generation
I use make_classification from sklearn.datasets to generate a synthetic dataset with 7 features and 1000 samples.

```
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=7, random_state=42)
We split the dataset into training, validation, and testing sets: ```

```
from sklearn.model_selection import train_test_split 

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) ```

## Building Models
Basic Model
The basic model uses a straightforward architecture with two hidden layers.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

basic_model = Sequential([
    Input(shape=(7,)),  
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])
basic_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Optimized Model
The optimized model introduces regularization and dropout layers to improve generalization. ```

```
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

optimized_model = Sequential([
    Dense(64, activation='relu', input_shape=(7,), kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
optimized_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) ```

## Training the Models
I train both models for 20 epochs using the training and validation data.

```
basic_history = basic_model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
optimized_history = optimized_model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val)) ```

## Evaluating the Models
I evaluate the trained models on the test dataset.

```
basic_eval = basic_model.evaluate(X_test, y_test, verbose=1)
optimized_eval = optimized_model.evaluate(X_test, y_test, verbose=1)

print(f"Basic Model - Loss: {basic_eval[0]}, Accuracy: {basic_eval[1]}")
print(f"Optimized Model - Loss: {optimized_eval[0]}, Accuracy: {optimized_eval[1]}") ```

## Saving the Models
I save the trained models for future use.

```
basic_model.save('saved_models/basic_model.h5')
optimized_model.save('saved_models/optimized_model.h5') ```

## Conclusion
This project demonstrates how model performance can be improved through techniques such as dropout and regularization. The optimized model, with these enhancements, is expected to show better generalization on unseen data.



