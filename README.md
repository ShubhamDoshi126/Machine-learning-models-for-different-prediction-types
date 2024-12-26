# Machine Learning Models Project

A comprehensive machine learning project demonstrating binary classification, multiclass classification, and regression using TensorFlow and Keras.

## Overview

This project showcases three different machine learning implementations:
- IMDb sentiment analysis (Binary Classification)
- Reuters news categorization (Multiclass Classification)
- Boston Housing price prediction (Regression)

## Project Structure

```
.
├── imdb.py            # IMDb binary classification implementation
├── reuters.py         # Reuters multiclass classification implementation
├── boston_housing.py  # Boston Housing regression implementation
└── module11.ipynb     # Training and evaluation notebook
```

## Features

**Data Processing**
- Automated data normalization
- Sequence vectorization
- Validation set creation
- Cross-validation implementation

**Model Types**
- Binary Classification (IMDb)
- Multiclass Classification (Reuters)
- Regression (Boston Housing)

## Installation

```bash
pip install tensorflow numpy matplotlib pandas
```

## Usage

### IMDb Sentiment Analysis

```python
from imdb import IMDb

# Initialize and train
imdb = IMDb()
imdb.prepare_data()
imdb.build_model()
imdb.train(epochs=20, batch_size=512)

# Evaluate
imdb.evaluate()
```

### Reuters Classification

```python
from reuters import Reuters

# Initialize and train
reuters = Reuters()
reuters.prepare_data()
reuters.build_model()
reuters.train(epochs=20, batch_size=512)

# Evaluate
reuters.evaluate()
```

### Boston Housing Regression

```python
from boston_housing import BostonHousing

# Initialize and train
boston = BostonHousing()
boston.prepare_data()
boston.build_model()
boston.train()

# Evaluate
boston.evaluate()
```

## Model Architectures

**IMDb Model**
- Dense neural network
- Binary cross-entropy loss
- Sigmoid activation

**Reuters Model**
- Multi-layer neural network
- Categorical cross-entropy loss
- Softmax activation

**Boston Housing Model**
- Regression neural network
- Mean squared error loss
- K-fold cross-validation

## Visualization

Each model includes methods for visualizing:
- Training/validation loss
- Accuracy metrics (classification models)
- Cross-validation results (regression model)

## Performance Monitoring

```python
# Plot training metrics
model.plot_loss()
model.plot_accuracy()  # For classification models
```
