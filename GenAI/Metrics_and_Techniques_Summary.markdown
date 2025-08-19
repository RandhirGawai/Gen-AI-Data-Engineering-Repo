# Simple Summary of Classification and Regression Metrics, Regularization, and Optimizers

## Classification Metrics
These metrics help evaluate how well a classification model performs.

| Metric | Formula | Use When | Notes |
|--------|---------|----------|-------|
| **Accuracy** | (Correct Positives + Correct Negatives) / Total | Classes have similar sizes | Not good if classes are uneven |
| **Error Rate** | 1 − Accuracy | Same as accuracy | Opposite of accuracy |
| **Precision** | Correct Positives / (Correct Positives + Flase Positives) | False positives are bad (e.g., spam detection) | Focuses on positive predictions |
| **Recall** | Correct Positives / (Correct Positives + False Negative) | Missing positives is bad (e.g., medical diagnosis) | Catches true positives |
| **Specificity** | Correct Negatives / (Correct Negatives + Wrong Negatives) | Need to correctly identify negatives | Often used with recall |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance precision and recall | Good for uneven classes |
| **Fβ Score** | Adjusts precision/recall balance | Want to focus on recall (β>1) or precision (β<1) | β=2 favors recall; β=0.5 favors precision |
| **G-Mean** | √(Recall × Specificity) | Uneven classes; care about both classes | Balances positive and negative performance |
| **Balanced Accuracy** | (Recall + Specificity) / 2 | Uneven classes | Handles class imbalance well |
| **Matthews Correlation** | Complex formula for agreement | Uneven or multi-class problems | Ranges from −1 (bad) to +1 (perfect) |
| **Cohen’s Kappa** | Agreement beyond chance | Noisy labels or multiple annotators | Measures true agreement |
| **ROC-AUC** | Area under curve (Recall vs. False Positive Rate) | Binary or one-vs-rest problems | Not affected by class imbalance; 0.5 = random, 1 = perfect |

# Precision vs. Recall: Easy Examples

## Easy Example 1: Spam Filter
- **False Positive**: Important email marked as spam ❌
  - *Bad!* You might miss a job offer.
- **Focus**: Precision

## Easy Example 2: Disease Test
- **False Negative**: Person has cancer, but test says “No” ❌
  - *Very dangerous!*
- **Focus**: Recall

## ✨ Summary in One Line
- 🎯 If saying “yes” by mistake is bad, focus on **precision**.
- 🔍 If missing a real case is bad, focus on **recall**.

## Regression Metrics
These metrics measure how close predictions are to actual values in regression tasks.

| Metric | Description | Use When | Notes |
|--------|-------------|----------|-------|
| **Mean Absolute Error (MAE)** | Average of absolute differences | Want simple error measure | Easy to understand |
| **Mean Squared Error (MSE)** | Average of squared differences | Larger errors matter more | Sensitive to outliers |
| **Root Mean Squared Error (RMSE)** | Square root of MSE | Want errors in original units | Easier to interpret than MSE |
| **Mean Absolute Percentage Error (MAPE)** | Average percentage error | Relative errors matter | Can be tricky with small values |
| **Symmetric MAPE (SMAPE)** | Adjusted percentage error | Avoid issues with small values | More stable than MAPE |
| **R-squared (R²)** | How much variance is explained | Check model fit | Can mislead in nonlinear cases |
| **Adjusted R²** | R² adjusted for number of features | Compare models with different features | Penalizes unnecessary features |
| **Median Absolute Error** | Median of absolute differences | Outliers are common | Less affected by outliers |
| **Huber Loss** | Mix of MAE and MSE | Some outliers expected | Adjustable with δ parameter |
| **Mean Bias Deviation (MBD)** | Average prediction error | Check for consistent over/under-prediction | Positive = under-predict, Negative = over-predict |
| **Mean Absolute Scaled Error (MASE)** | Error scaled by simple forecast | Time series data | Works across different scales |

## Regularization Techniques in Deep Learning
These methods help prevent models from overfitting.

| Technique | What It Does | Use When | Notes |
|-----------|--------------|----------|-------|
| **L1 (Lasso)** | Adds penalty for weight size | Need sparse models | Sets some weights to zero |
| **L2 (Ridge)** | Adds penalty for squared weight size | Mild overfitting | Keeps weights small |
| **ElasticNet** | Combines L1 and L2 | Need sparsity and small weights | Balances both methods |
| **Dropout** | Randomly turns off neurons | Deep networks; small datasets | Common rate: 0.2–0.5 |
| **Batch Normalization** | Standardizes layer inputs | Faster training, better results | Also acts as regularization |
| **Early Stopping** | Stops when validation error stops improving | Monitor validation loss | Simple way to avoid overfitting |
| **Data Augmentation** | Creates new training data (e.g., rotated images) | Image, text, or audio data | Improves model generalization |
| **Label Smoothing** | Softens confident predictions | Noisy labels or overconfident models | Helps with model confidence |

# Cross-Validation in Machine Learning

## Overview
Cross-validation is a technique to check how well a machine learning model will perform on new, unseen data. It’s like giving a student practice tests before the final exam to see how ready they are.

## Simple Analogy
Imagine you’re a teacher preparing students for a final exam. Instead of giving just one practice test, you:
- Create multiple practice tests using different parts of the study material.
- Each time, hide some questions and test the students on those.
- This gives a better idea of how they’ll do on the real exam.

## How Cross-Validation Works
**Basic Idea**: Instead of splitting your data into one training and testing set, you split it multiple times in different ways to test the model thoroughly.

### Most Common Type - K-Fold Cross-Validation
- Split your data into **K equal parts** (usually 5 or 10).
- Use **K-1 parts for training** and **1 part for testing**.
- Repeat **K times**, using a different part for testing each time.
- Average the results from all K tests to get a reliable performance score.

### Example with 5-Fold Cross-Validation
If you have 1000 data points:
- **Round 1**: Train on data 1–800, test on data 801–1000.
- **Round 2**: Train on data 1–600 + 801–1000, test on data 601–800.
- **Round 3**: Train on data 1–400 + 601–1000, test on data 401–600.
- **Round 4**: Train on data 1–200 + 401–1000, test on data 201–400.
- **Round 5**: Train on data 201–1000, test on data 1–200.
- Average the accuracy from all 5 rounds to estimate model performance.

## Why Use Cross-Validation?
- **More Reliable Results**: A single train-test split might be lucky or unlucky. Multiple splits give a better average.
- **Prevents Overfitting**: If a model only works well on one test set, cross-validation will reveal this issue.
- **Better Use of Data**: Every data point is used for both training and testing at some point.

## Types of Cross-Validation
- **K-Fold**: Splits data into K parts; most common method.
- **Leave-One-Out**: Trains on all data except one point; good for small datasets but slow.
- **Stratified**: Ensures each fold has the same proportion of classes (e.g., spam vs. non-spam).
- **Time Series**: Respects chronological order for time-based data.

## When to Use Cross-Validation
- When you have **limited data** and want to use it efficiently.
- When you need a **reliable estimate** of model performance.
- Before **deploying a model** to ensure it works well in production.
- When **comparing models** or algorithms to choose the best one.
# Regularization Techniques: L1, L2, and Elastic Net

## When to Use L1 (Lasso)?

| **Situation** | **Why L1 Helps** |
|---------------|------------------|
| You have a lot of features, but think only a few matter | L1 removes unimportant features by setting their weights to exactly 0. |
| You want to do feature selection automatically | L1 picks only the most useful features. |

**Example**:  
Predicting house prices with 1000 features (e.g., zip code, school ratings, etc.), but only 10 are truly important → use L1.

## When to Use L2 (Ridge)?

| **Situation** | **Why L2 Helps** |
|---------------|------------------|
| You have many features, and all might be useful a little | L2 shrinks weights but keeps all features. |
| You want to avoid overfitting without eliminating any feature | L2 keeps weights balanced to improve generalization. |

**Example**:  
Predicting temperature using readings from many sensors, where all contribute a bit → use L2.

## Can I Use Both?

Yes! This is called **Elastic Net**:
- Combines L1 and L2 regularization.
- Use when:
  - You want feature selection (from L1).
  - You want stable generalization (from L2).

## Summary

| **Use Case** | **Regularization** |
|--------------|---------------------|
| Want to drop irrelevant features | L1 (Lasso) |
| Want to shrink weights smoothly | L2 (Ridge) |
| Want both benefits | Elastic Net |



## Optimizers in Deep Learning
These algorithms update model weights to improve performance.

| Optimizer | How It Works | Use When | Notes |
|-----------|--------------|----------|-------|
| **SGD** | Simple gradient-based updates | Basic problems; stable gradients | Slow; needs tuning |
| **Momentum** | Adds speed to gradient updates | Deep networks; smoother updates | Reduces zig-zagging |
| **NAG** | Looks ahead before updating | Improves momentum | Combines lookahead and momentum |
| **Adagrad** | Adjusts learning rate per feature | Sparse data (e.g., text) | Learning rate shrinks over time |
| **RMSprop** | Uses moving average of gradients | Recurrent networks; online learning | Avoids vanishing learning rates |
| **Adam** | Combines momentum and RMSprop | Most deep learning tasks | Fast and widely used |
| **AdamW** | Adam with better weight decay | Transformers, large models | Better regularization |
| **AdaDelta / AdaMax** | Adaptive learning rate alternatives | Adam underperforms | Niche use cases |
| **Lookahead** | Combines fast and slow updates | Fine-tuning stability | Works with other optimizers |
| **LAMB / Lion** | Optimized for large batches | Large models (e.g., BERT, GPT) | Good for huge datasets |
