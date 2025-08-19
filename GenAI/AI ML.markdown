# Data Science Notes


# Checklist to Diagnose a Non-Performing Chatbot

## 1. Prompt Design Issues
- **Prompt too vague or too long?**  
  Prompts may lack clarity, miss key instructions, or exceed token limits.
- **Missing system prompt or role?**  
  Clearly define the system’s behavior (e.g., “You are a helpful assistant using only verified information”).
- **Prompt formatting error?**  
  Check for improper spacing, missing delimiters, or malformed input.

## 2. Model-Related Issues
- **Wrong model selected?**  
  Using a smaller or less capable model (e.g., GPT-3.5 instead of GPT-4 or a domain-specific model).
- **Context window exceeded?**  
  If input, retrieved documents, and prompt exceed token limits, important context may be cut off.
- **Temperature too high?**  
  High temperature makes responses more creative but less accurate. Use 0.2–0.5 for factual tasks.

## 3. RAG Pipeline Issues (Retrieval-Augmented Generation)
- **Bad document chunking?**  
  Chunks may be too small (losing context) or too large (exceeding limits).
- **Irrelevant document retrieval?**  
  Verify if vector embeddings return the correct top-k documents.
- **Embedding mismatch?**  
  Ensure query and documents use the same embedding model or tokenizer.
- **Stale or incomplete knowledge base?**  
  Check if documents are outdated or missing key sources.

## 4. Data Quality Issues
- **Noisy or unstructured source documents?**  
  PDFs or scraped text may contain errors or misformatted sections.
- **Incomplete ingestion?**  
  Ensure the pipeline processes and stores all data correctly.
- **OCR errors (for scanned files)?**  
  Poor text extraction from images or PDFs can reduce quality.

## 5. Retrieval Engine Issues
- **Incorrect top-k selection?**  
  Top-k may be too low or return irrelevant chunks due to poor similarity scores.
- **Wrong similarity function?**  
  Cosine, dot product, or Euclidean distance can affect relevance ranking.
- **No semantic search used?**  
  Keyword-based search may be used instead of vector-based retrieval.

## 6. Backend/API Issues
- **Model API returning errors silently?**  
  Check for timeouts, token limits, rate limits, or unhandled HTTP errors.
- **Missing or wrong input fields?**  
  Ensure the backend sends complete and correct JSON or prompt structures.
- **Retries not handled?**  
  Failed responses should trigger retries or fallbacks.

## 7. User Query Understanding
- **Query unclear or ambiguous?**  
  Add a layer to ask clarifying questions or use semantic parsing.
- **Misspelled or multilingual query?**  
  Include spell correction or translation preprocessing if needed.
- **Too broad or too narrow queries?**  
  Example: “Tell me about banking” vs. “What are RBI guidelines for UPI in 2024?”

## 8. Evaluation & Logging
- **No logging of inputs/outputs?**  
  Log prompts, responses, retrieved documents, and user feedback for analysis.
- **Lack of feedback loop?**  
  Add thumbs up/down or a way for users to flag bad responses to improve the system.


## Data Engineering / Pipelines

### Full Lifecycle of a Data Pipeline
The lifecycle of a data pipeline involves multiple stages to ensure data is ingested, transformed, stored, and made available for analysis or modeling.

**Example Implementation**:
- **Source Ingestion**: Streamed data using AWS IoT Core into AWS Kinesis for real-time data capture.
- **ETL (Extract, Transform, Load)**: AWS Glue with PySpark transformed JSON data, partitioned it by date, and stored it in Parquet format in S3.
- **Storage and Query**: Used Amazon Athena for querying the Parquet data, with data accessible to SageMaker for ML model training.
- **Monitoring**: Configured CloudWatch for logging and alerting to ensure pipeline reliability.

## Machine Learning Modeling & Optimization



### Random Search vs. Grid Search
# Model Hyperparameters

| Model Type           | Hyperparameters Examples                           |
|----------------------|---------------------------------------------------|
| Linear Models        | Regularization (`alpha`, `lambda`)                 |
| Decision Trees       | `max_depth`, `min_samples_split`                  |
| XGBoost/RandomForest | `n_estimators`, `max_depth`, `learning_rate`      |
| Neural Networks      | `learning_rate`, `batch_size`, `epochs`, `dropout` |
- **Random Search**: Randomly samples hyperparameter combinations.
  - **Advantages**:
    - Faster for large search spaces.
    - Effective when some parameters have less impact.
    - Requires fewer iterations for good results.
  - **Preference**: Use Random Search when computational resources are limited or when exploring a large hyperparameter space.

### Optimization Techniques for Model Training
- Early stopping to prevent overfitting.
- Learning rate scheduling for adaptive step sizes.
- Regularization (L1, L2) to reduce model complexity.
- Gradient clipping for stable deep learning training.
- Hyperparameter tuning using Optuna for efficient search.

# Boosting Techniques: Simple Examples

## AdaBoost (Adaptive Boosting) - Simple Example

Let’s say we’re building a spam filter for emails. We have four emails:

- **Email 1**: "Buy now!" → **SPAM**
- **Email 2**: "Meeting at 3pm" → **NOT SPAM**
- **Email 3**: "Free money!!!" → **SPAM**
- **Email 4**: "How are you?" → **NOT SPAM**

### Step 1: Give Equal Attention to All Emails
- Each email gets a weight of 25% (equal importance).
- **Model 1 (simple rule)**: "If email has exclamation marks, it’s spam."
- **Predictions**: SPAM, NOT SPAM, SPAM, NOT SPAM.
- **Result**: All correct! (But this is often not the case.)

### Let’s Say Model 1 Made Mistakes
- **Model 1 predicted**: SPAM, **SPAM** (wrong!), SPAM, NOT SPAM.
- Email 2 was misclassified (predicted SPAM, but it’s NOT SPAM).

### Step 2: Pay More Attention to Mistakes
- Email 2 gets a higher weight (e.g., 40% instead of 25%).
- Other emails get lower weights (e.g., 20% each).
- This means "Email 2 is more important to get right next time!"

### Step 3: Train Model 2 with New Weights
- **Model 2 (new rule)**: "If email is very short, it’s not spam."
- Because Email 2 has higher weight, Model 2 focuses on getting it right.
- **Model 2 predicts**: Email 2 is NOT SPAM (correct).

### Step 4: Combine Models with Voting
- Final prediction = Weighted vote from all models.
- Model 1 gets a vote weight based on its performance.
- Model 2 gets a vote weight based on its performance.
- Better models have a louder voice in the final decision.

### Key Point
AdaBoost is like a teacher who:
- Notices which students (data points) are struggling.
- Spends extra time helping those students.
- Tracks which teaching methods (models) work best.
- Gives more importance to the best teachers’ opinions.

---

## Gradient Boosting - Simple Example

Let’s say we want to predict house prices. We have three houses with actual sale prices:
- House 1: $300K
- House 2: $250K
- House 3: $400K

### Step 1: Start Simple
- **Model 1**: Predicts the average of all house prices.
- Average = (300 + 250 + 400) ÷ 3 = $317K.
- **Model 1 predicts**: $317K, $317K, $317K for all houses.

### Step 2: Look at the Mistakes (Residuals)
- House 1: Actual $300K - Predicted $317K = **-$17K** (overestimated).
- House 2: Actual $250K - Predicted $317K = **-$67K** (overestimated).
- House 3: Actual $400K - Predicted $317K = **+$83K** (underestimated).
- These mistakes are called "residuals."

### Step 3: Train a New Model to Predict These Mistakes
- **Model 2** learns to predict: -$17K, -$67K, +$83K.
- Let’s say Model 2 predicts: -$10K, -$50K, +$60K.

### Step 4: Combine the Models
- Final Prediction = Model 1 + Model 2:
  - House 1: $317K + (-$10K) = **$307K** (closer to $300K).
  - House 2: $317K + (-$50K) = **$267K** (closer to $250K).
  - House 3: $317K + (+$60K) = **$377K** (closer to $400K).

### Step 5: Still Have Mistakes? Add Another Model
- New residuals after Model 1 + Model 2:
  - House 1: $300K - $307K = **-$7K**.
  - House 2: $250K - $267K = **-$17K**.
  - House 3: $400K - $377K = **+$23K**.
- **Model 3** learns to predict: -$7K, -$17K, +$23K.
- Keep adding models until mistakes are tiny.

### Key Insight
Instead of predicting house prices directly, we:
- Start with a simple guess.
- Look at what we got wrong (residuals).
- Train a new model to fix those mistakes.
- Add the corrections to the original guess.
- Repeat until predictions are very accurate.

### Why This Works
- Each model has an easier job: fix the remaining errors.
- Models specialize (e.g., one for expensive houses, another for cheap ones).
- Gradual improvement makes predictions better each step.
- Think of it like painting a wall: the first coat covers most of it, the second fixes missed spots, and the third handles tiny imperfections.

---

## XGBoost - Additional Features

XGBoost builds on Gradient Boosting with extra features to make it faster and more robust:

### Superpower 1: Regularization (Prevents Overfitting)
- **Problem**: Models might memorize training data but fail on new data.
- **Solution**: Penalizes complex models, like a student who understands concepts instead of memorizing answers.
- **Result**: Better performance on unseen data.

### Superpower 2: Parallel Processing (Speed)
- **Problem**: Gradient Boosting is slow on large datasets.
- **Solution**: Uses multiple computer cores simultaneously, like eight people building a house instead of one.
- **Result**: Up to 10x faster training.

### Superpower 3: Smart Missing Data Handling
- **Problem**: Missing data (e.g., no garage info for a house).
- **Solution**: Learns the best way to handle missing values, like a GPS finding routes around closed roads.
- **Result**: No need for manual preprocessing.

### Superpower 4: Tree Pruning (Smart Stopping)
- **Problem**: Decision trees may have unnecessary branches.
- **Solution**: Builds a full tree, then cuts off useless branches, like editing out unnecessary parts of an essay.
- **Result**: Simpler, better-performing models.

### Superpower 5: Built-in Cross-Validation
- **Problem**: How do you know if your model is good?
- **Solution**: Automatically tests on held-out data, like taking practice tests while studying.
- **Result**: Reliable performance estimates.

### Superpower 6: Early Stopping
- **Problem**: Training too long can cause overfitting.
- **Solution**: Stops training when performance stops improving, like stopping study when you’re not learning more.
- **Result**: Optimal training without manual intervention.
- **Example**:
  - **Data**: House prices with features (e.g., 2 BHK, 900 sqft → 50 lakhs).
  - **Process**: Predict base value (50 lakhs), compute residuals (e.g., +20, -20), build trees to correct errors, and refine predictions.

- **Key Parameters**:
  - `n_estimators`: Number of trees.
  - `max_depth`: Depth of each tree.
  - `learning_rate`: Step size for updates.
  - `subsample`: Percentage of rows used per tree.
  - `colsample_bytree`: Percentage of columns used per tree.
  - `reg_alpha (L1)`: Lasso regularization.
  - `reg_lambda (L2)`: Ridge regularization.

- **Use Cases**:
  - Structured/tabular data.
  - Datasets with missing values.
  - High-accuracy tasks (e.g., Kaggle competitions).

### Optimization Algorithms in ML/DL
Optimization algorithms minimize the loss function by updating model weights.

- **Common Optimizers**:
  - **Gradient Descent (GD)**: Updates weights using all data (slow but stable).
  - **Stochastic Gradient Descent (SGD)**: Updates per data point (fast but noisy).
  - **Mini-batch GD**: Uses small batches (balanced, commonly used).
  - **Momentum**: Accelerates updates using past gradients.
  - **RMSprop**: Adapts learning rate per parameter based on recent gradients.
  - **Adam**: Combines Momentum and RMSprop (widely used in deep learning).

- **Gradient Descent Process**:
  1. Initialize weights randomly.
  2. Compute gradient of loss with respect to weights.
  3. Update weights to reduce loss.
  4. Repeat until convergence.

- **Batch vs. Stochastic vs. Mini-batch GD**:
  - **Batch GD**: Uses all data per update (e.g., 1000 samples).
  - **SGD**: Updates per sample (fast but unstable).
  - **Mini-batch GD**: Uses small batches (e.g., 32 or 64 samples) for balance.

- **Early Stopping**:
  - **Purpose**: Stops training when validation performance degrades to prevent overfitting.
  - **Example**: Stop at epoch 40 if validation accuracy drops, even if training runs to 100 epochs.

- **Regularization Techniques**:
  - **L1 (Lasso)**: Adds absolute weight penalty, promotes sparsity (feature selection).
  - **L2 (Ridge)**: Adds squared weight penalty, shrinks weights (keeps all features).
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

- **Learning Rate**:
  - **Definition**: Controls step size for weight updates.
  - **Tuning**: Start with 0.01, 0.001, or 0.0001; use schedulers or experiments to optimize.
  - **Impact**: Too high → overshoots; too low → slow convergence.

- **Adam Optimizer**:
  - Combines Momentum (past gradients) and RMSprop (adaptive learning rates).
  - **Use Case**: Deep learning, noisy data, sparse gradients.
  - **Example**: Preferred over SGD for faster convergence in neural networks.

- **Gradient Clipping**:
  - **Purpose**: Limits gradient size to prevent exploding gradients in deep networks.
  - **Example**: Clip gradients to 5.0 to avoid NaN loss in LSTMs.

- **Handling Overfitting**:
  - Early stopping, L1/L2 regularization, dropout, data augmentation, reduce model complexity, cross-validation.
  - **Example**: Apply dropout and image augmentation for better generalization in image classification.

- **Model Checkpointing**:
  - **Purpose**: Save model at peak validation performance to avoid retraining or using suboptimal models.
  - **Example**: Save model at epoch 25 if it has the highest validation accuracy.

## Data Analysis & Visualization

### Exploratory Data Analysis (EDA) Steps
- Check and impute missing values.
- Detect outliers using IQR or Z-score.
- Analyze multicollinearity with a correlation matrix.
- Visualize distributions with histograms or KDE plots.
- Encode and balance categorical features.
- Select features based on importance scores.

### Visualization Techniques
- **Categorical Data**: Bar plots, countplots.
- **Numerical Data**: Histograms, boxplots, KDE plots.
- **Correlation**: Heatmaps.
- **Time Series**: Line plots.
- **Multivariate**: Pairplots, scatter matrices.

## Statistical Analysis

### Hypothesis Testing in Real-World Projects
# Hypothesis Testing

## Overview
Hypothesis testing is a statistical method used to make decisions or inferences about a population based on sample data. It helps answer questions like: *“Did the new model, layout, or campaign actually make a difference, or was it just random chance?”*

### Real-World Example: A/B Test for Website Layout
Consider an A/B test comparing two website layouts:
- **Group A**: Sees the old website layout.
- **Group B**: Sees the new layout.
- **Question**: Does the new layout improve the conversion rate?

## Step-by-Step Breakdown
1. **Define Hypotheses**
   - **Null Hypothesis (H₀)**: The new layout has no effect (conversion rates are the same).
   - **Alternative Hypothesis (H₁)**: The new layout improves conversion rates.
2. **Choose Significance Level**
   - Typically, `α = 0.05` (5% chance of incorrectly rejecting H₀).
3. **Run the Experiment**
   - Show Layout A to 50% of users and Layout B to the other 50%. Collect conversion data.
4. **Choose a Test**
   - Numerical and normal data: Use a **t-test**.
   - Categorical data: Use a **chi-square test**.
5. **Calculate p-value**
   - The p-value indicates the probability that the observed difference occurred by random chance.
6. **Make Decision**
   - If `p < 0.05`: Reject H₀, concluding the new layout significantly improves conversions.
   - If `p ≥ 0.05`: Fail to reject H₀, indicating no significant improvement.

## Techniques Used in Hypothesis Testing (with ML Relevance)
| Technique                  | Use Case                                    | Example                                      |
|----------------------------|---------------------------------------------|----------------------------------------------|
| **t-test**                | Compare means of two groups                 | Conversion rate in A/B testing               |
| **z-test**                | Like t-test, for large sample sizes         | Click-through rates comparison              |
| **ANOVA**                 | Compare means of three or more groups       | Testing three different ad campaigns         |
| **Chi-square test**       | Compare categorical distributions           | Gender distribution in model predictions    |
| **Mann–Whitney U**        | Non-parametric version of t-test           | When data is not normally distributed       |
| **Kolmogorov–Smirnov**    | Check if two samples come from same distribution | Model prediction distributions          |
| **Permutation test**      | Shuffle labels to build null distribution   | ML model score significance                 |

## Importance of Hypothesis Testing in ML Projects
| Scenario                        | Hypothesis Testing Use                     |
|---------------------------------|--------------------------------------------|
| **Model Comparison**            | Test if a new model's accuracy improvement is significant. |
| **Feature Impact**              | Assess whether a new feature adds value to the model. |
| **Marketing/UX A/B Tests**      | Evaluate if a new campaign or design improves KPIs. |
| **Data Quality Checks**         | Detect unexpected changes in data distributions. |
| **Bias Detection**              | Check fairness across demographic groups. |

## Generative AI, Embeddings & RAG

### Embedding Models
- **Used**: HuggingFace’s `all-MiniLM-L6-v2` for RAG semantic search (fast, balanced performance).
- **Alternative**: `BAAI/bge-base-en-v1.5` or OpenAI embeddings for higher quality in GenAI tasks.

### Retrieval Techniques
- **Exact Match**: BM25 for keyword-based search.
- **Vector Similarity**: FAISS with cosine similarity for semantic search.
- **MMR (Maximal Marginal Relevance)**: Reduces redundancy, increases diversity in retrieved results.

### Handling Image Data in Pipelines
- **Preprocessing**: Use PIL or OpenCV for resizing and normalization.
- **Embedding Generation**: Convert images to embeddings with CLIP or ResNet50.
- **Storage**: Store embeddings in FAISS; metadata and image paths in a database.
- **Example**: Preprocess images, generate embeddings, and retrieve similar images using FAISS.

### Evaluating RAG Models
- **Metrics**: Use RAGAS or custom metrics like Context Precision, Faithfulness, and Answer Relevance.
- **Approach**: Measure quality of retrieved context and generated responses.

## Microsoft Copilot & Power Platform

### Microsoft Copilot vs. Power Platform
| Feature            | Microsoft Copilot                            | Power Platform                              |
|--------------------|---------------------------------------------|---------------------------------------------|
| **Purpose**        | AI assistant for Microsoft 365 apps         | Low-code platform for apps and workflows    |
| **User Interaction**| Chat-style in Word, Excel, etc.             | Drag-and-drop UI with logic flows           |
| **Tech Stack**     | OpenAI + Microsoft Graph                    | Power Apps, Power Automate, etc.            |
| **Customization**  | Limited to Microsoft apps                   | Highly customizable workflows               |

### Pipelines in Copilot-Based Solutions
- Data ingestion and preprocessing.
- Embedding generation (text, image).
- Vector store integration (e.g., Azure AI Search).
- Retrieval and reranking (MMR or dense retrieval).
- Prompt construction for LLMs.
- LLM generation (e.g., OpenAI, Phi-3).
- Post-processing and response generation.

## MLOps and Advanced Topics

### MLOps Approach
- **Versioning**: Use MLflow for model and data versioning.
- **Pipeline Automation**: Use Airflow for scheduling and orchestrating pipelines.
- **Monitoring**: Track model performance and data drift.
- **CI/CD**: Automate model deployment and testing.

### L1 vs. L2 Regularization
- **L1 (Lasso)**: Adds absolute weight penalty, promotes sparsity (feature selection).
- **L2 (Ridge)**: Adds squared weight penalty, shrinks weights (keeps all features).
- **Example**: Use L1 to eliminate irrelevant features, L2 to balance feature contributions.

## Handling Missing Data

### Types of Missing Data
- **MCAR (Missing Completely At Random)**:
  - No pattern; unrelated to features or target.
  - **Example**: Random sensor failures.
  - **Action**: Drop or simple imputation (mean, median).
- **MAR (Missing At Random)**:
  - Missingness depends on other features.
  - **Example**: Income missing for younger people.
  - **Action**: Impute using related columns (e.g., age).
- **MNAR (Missing Not At Random)**:
  - Missingness depends on the value itself.
  - **Example**: High-income individuals not disclosing income.
  - **Action**: Use domain knowledge or advanced models.

### Imputation Methods
| Method             | Description                       | Use Case                   |
|--------------------|-----------------------------------|----------------------------|
| Mean/Median        | Replace with average              | Numerical data             |
| Mode               | Use most frequent value           | Categorical data           |
| Constant Value     | Fill with fixed value (e.g., 0)   | Both numerical/categorical |
| Forward/Backward Fill | Use previous/next value        | Time-series data           |
| Drop Rows          | Remove rows with missing values   | Rare missing data          |

### Handling Categorical vs. Numerical Data
- **Numerical**: Use mean, median, or KNN imputation; prefer median for skewed data.
  - **Example**: Impute missing age with median (e.g., 30).
- **Categorical**: Use mode or placeholder like "Unknown."
  - **Example**: Impute missing gender with mode (e.g., "Male") or "Unknown."

### KNN Imputation
- **Process**: Find similar rows (neighbors) and impute based on their values.
- **Example**: Impute missing student marks using marks of similar students (based on age, grade).
- **Advantage**: More accurate than mean but computationally expensive.

### Iterative/Multivariate Imputation
- **Process**: Predict missing values using regression models iteratively across features.
- **Example**: Predict missing age using a model trained on gender, salary, and experience.
- **Tool**: Scikit-learn’s `IterativeImputer`.

### When to Drop vs. Impute
| Situation                              | Action                              |
|----------------------------------------|-------------------------------------|
| Few missing rows                       | Drop rows                           |
| Many missing rows                      | Impute to avoid data loss           |
| Feature >40% missing                   | Consider dropping column            |
| MCAR                                   | Drop or simple imputation           |
| MAR/MNAR                               | Advanced imputation (e.g., KNN)     |

### Impact of Imputation on Model Accuracy
- **Good Imputation**: Preserves data relationships, improves accuracy.
- **Poor Imputation**:
  - Adds bias (e.g., filling income with 0).
  - Reduces variance or misleads the model.
- **Best Practice**: Test multiple imputation methods and validate with cross-validation.
