## 1. Project Overview

### Problem Statement

The objective of this challenge is to classify multivariate time-series data related to various pirates to determine their specific level of perceived pain. The task requires labeling each sequence as **no_pain**, **low_pain**, or **high_pain**. Success is measured primarily by the F1-score, which balances precision and recall.

## 2. Data Preparation

The dataset contains recordings for 661 pirates in the training set and 1,324 in the test set.

* **Feature Structure**: Each pirate is linked to 160 data points, each containing 40 features.

* **Data Types**: The features include temporal indices, four integer-based pain surveys, categorical subject traits (number of legs, hands, and eyes), and 31 real-valued body joint angles.

## 3. Model Development

* **Core Architecture**: The model utilizes a **Bidirectional LSTM** framework to process temporal dependencies.

* **Feature Handling**: An **Embedding layer** specifically manipulates categorical data (n_legs, n_hands, n_eyes) to improve performance.

* **Loss Function**: A **Weighted Cross-Entropy loss** is implemented to give higher importance to less frequent classes in the training set.

* **Regularization**: Overfitting is controlled through **Dropout** (20% probability) and **Early Stopping** with a 50-epoch patience.

* **Output Layer**: A final **Softmax layer** allows the model to output probabilities for each of the three pain classes.

* **Logic Correction**: A critical fix is applied to the **majority voting** formula, which originally grouped pirate sequences incorrectly and caused poor initial test results.


## 4. Results

The final model achieves high performance and demonstrates strong generalization capabilities.

The most significant performance gains come from the Embedding layer, the Weighted loss, and Majority Voting implementation.
