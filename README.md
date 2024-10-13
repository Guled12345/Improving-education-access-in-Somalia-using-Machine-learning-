
# Improving Education Access in Somalia Using Machine Learning

## Project Overview
This project aims to create a predictive machine learning model that identifies students in Somalia at risk of dropping out or receiving insufficient education due to socioeconomic or geographical factors.

## Dataset Generation
I use `make_classification` from `sklearn.datasets` to generate a synthetic dataset with 7 features and 1000 samples.

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=7, random_state=42) ```

# I split the dataset into training, validation, and testing sets:


``` from sklearn.model_selection import train_test_split
 X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) ```

 # Model Architectures
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
basic_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) ```
 
 # Optimized Model
The optimized model introduces regularization and dropout layers to improve generalization.

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
 
 ## Error Analysis
After training our models, I performed an error analysis to understand how well the models generalize. We used the following methods:

Confusion Matrix: Helps visualize the performance of the model.
Classification Report: Provides precision, recall, F1-score, and accuracy metrics.
Misclassified Samples: Identified samples where model predictions differed from actual labels.
Code for Error Analysis

```import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

y_pred = (basic_model.predict(X_test) > 0.5).astype("int32") ```

``` cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm) ```

``` report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)``` 

``` misclassified = np.where(y_test != y_pred.flatten())[0]
print(f"\nNumber of Misclassified Samples: {len(misclassified)}")
print(f"Indices of Misclassified Samples: {misclassified}") ``` 

## Conclusion
This project demonstrates how model performance can be improved through techniques such as dropout and regularization. The optimized model is expected to show better generalization on unseen data.

## How to Run
Project Setup
Before proceeding, ensure you have the necessary dependencies installed. You can install the required libraries using:

```bash
pip install tensorflow scikit-learn ```

I also recommend using Google Colab for this project to leverage GPU support.

 Clone the Repository
Clone the repository to your local machine:

bash
```
git clone https://github.com/your_username/improving-education-access-somalia ```
cd improving-education-access-somalia
 
 ## Create a Virtual Environment
It’s recommended to create a virtual environment to manage dependencies. You can use either venv or conda:
Feel free to customize it further to match your project specifics!
