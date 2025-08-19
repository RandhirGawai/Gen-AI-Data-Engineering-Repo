# NLP Questions and Answers with Coding Examples

## 1. Fundamental NLP Concepts

### Q1: What is Natural Language Processing (NLP)?
**Answer**: NLP is a branch of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language in a valuable way.

### Q2: What are the main challenges in NLP?
**Answer**:
- Ambiguity: Words can have multiple meanings
- Context dependency: Meaning changes based on context
- Syntax variations: Different ways to express the same idea
- Named entity recognition: Identifying proper nouns
- Sarcasm and sentiment: Understanding tone and emotion
- Language evolution: New words and expressions

### Q3: What is tokenization?
**Answer**: Tokenization is the process of breaking down text into smaller units called tokens (words, subwords, or characters).

**Code Example**:
```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

text = "Hello world! How are you today?"
word_tokens = word_tokenize(text)
sentence_tokens = sent_tokenize(text)

print("Word tokens:", word_tokens)
print("Sentence tokens:", sentence_tokens)
```

### Q4: What is stemming and lemmatization?
**Answer**:
- **Stemming**: Reduces words to their root form by removing suffixes (crude but fast)
- **Lemmatization**: Reduces words to their base dictionary form (more accurate but slower)

**Code Example**:
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "ran", "runs", "easily", "fairly"]

print("Stemming:")
for word in words:
    print(f"{word} -> {stemmer.stem(word)}")

print("\nLemmatization:")
for word in words:
    print(f"{word} -> {lemmatizer.lemmatize(word)}")
```

## 2. Text Preprocessing

### Q5: What are stop words and why do we remove them?
**Answer**: Stop words are common words (like "the", "is", "at") that usually don't carry significant meaning. They're removed to focus on important words and reduce computational complexity.

**Code Example**:
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
text = "This is a sample sentence with stop words"
tokens = word_tokenize(text)

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Original:", tokens)
print("Filtered:", filtered_tokens)
```

### Q6: What is Part-of-Speech (POS) tagging?
**Answer**: POS tagging is the process of assigning grammatical tags to words (noun, verb, adjective, etc.).

**Code Example**:
```python
import nltk
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

print("POS Tags:", pos_tags)
```

## 3. Feature Engineering

### Q7: What is Bag of Words (BoW)?
**Answer**: BoW is a text representation method that describes documents by word occurrence frequencies, ignoring grammar and word order.

**Code Example**:
```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "Machine learning is great",
    "I love programming"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:")
print(bow_matrix.toarray())
```

### Q8: What is TF-IDF?
**Answer**: TF-IDF (Term Frequency-Inverse Document Frequency) weighs terms by their frequency in a document and rarity across the corpus.

**Code Example**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love machine learning",
    "Machine learning is great",
    "I love programming"
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("Feature names:", tfidf_vectorizer.get_feature_names_out())
```

## 4. Word Embeddings

### Q9: What are word embeddings?
**Answer**: Word embeddings are dense vector representations of words that capture semantic relationships in a continuous vector space.

### Q10: What is Word2Vec?
**Answer**: Word2Vec is a neural network model that learns word embeddings using either Skip-gram or CBOW (Continuous Bag of Words) architecture.

**Code Example**:
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample corpus
sentences = [
    "I love machine learning",
    "Machine learning is fascinating",
    "Deep learning is a subset of machine learning",
    "Natural language processing uses machine learning"
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get word vector
try:
    vector = model.wv['machine']
    print("Vector for 'machine':", vector[:5])  # Show first 5 dimensions
    
    # Find similar words
    similar_words = model.wv.most_similar('machine', topn=3)
    print("Similar words to 'machine':", similar_words)
except KeyError:
    print("Word not in vocabulary")
```

## 5. Named Entity Recognition (NER)

### Q11: What is Named Entity Recognition?
**Answer**: NER is the task of identifying and classifying named entities (person, organization, location, etc.) in text.

**Code Example**:
```python
import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. is planning to open a new store in New York next month."
doc = nlp(text)

print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_} - {spacy.explain(ent.label_)}")
```

## 6. Sentiment Analysis

### Q12: What is sentiment analysis?
**Answer**: Sentiment analysis determines the emotional tone or opinion expressed in text (positive, negative, neutral).

**Code Example**:
```python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Using TextBlob
text = "I love this product! It's amazing."
blob = TextBlob(text)
print(f"TextBlob Sentiment: {blob.sentiment}")

# Using VADER
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = analyzer.polarity_scores(text)
print(f"VADER Sentiment: {sentiment_scores}")
```

## 7. Language Models

### Q13: What is an n-gram model?
**Answer**: An n-gram model predicts the next word based on the previous (n-1) words. It's a statistical language model.

**Code Example**:
```python
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text.lower())

# Generate bigrams and trigrams
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("Bigrams:", bigrams)
print("Trigrams:", trigrams)

# Count frequency
bigram_freq = Counter(bigrams)
print("Bigram frequencies:", bigram_freq)
```

### Q14: What is perplexity in language models?
**Answer**: Perplexity measures how well a probability model predicts a sample. Lower perplexity indicates better performance.

**Code Example**:
```python
import math

def calculate_perplexity(probabilities):
    """Calculate perplexity given a list of probabilities"""
    log_sum = sum(math.log2(p) for p in probabilities if p > 0)
    perplexity = 2 ** (-log_sum / len(probabilities))
    return perplexity

# Example probabilities for words in a sentence
word_probabilities = [0.1, 0.3, 0.2, 0.4]
perplexity = calculate_perplexity(word_probabilities)
print(f"Perplexity: {perplexity}")
```

## 8. Deep Learning in NLP

### Q15: What is an RNN and why is it useful for NLP?
**Answer**: RNNs (Recurrent Neural Networks) can process sequential data and maintain memory of previous inputs, making them suitable for text processing.

**Code Example**:
```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take last output
        return out

# Example usage
input_size = 100  # embedding dimension
hidden_size = 128
output_size = 2   # binary classification
model = SimpleRNN(input_size, hidden_size, output_size)
print(model)
```

### Q16: What is attention mechanism?
**Answer**: Attention allows models to focus on relevant parts of the input sequence when making predictions, improving performance on long sequences.

**Code Example**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(hidden_states), dim=1)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        return context_vector, attention_weights

# Example usage
hidden_size = 128
seq_len = 10
batch_size = 32

attention_layer = AttentionLayer(hidden_size)
hidden_states = torch.randn(batch_size, seq_len, hidden_size)
context, weights = attention_layer(hidden_states)
print(f"Context vector shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")
```

## 9. Transformers

### Q17: What is a Transformer architecture?
**Answer**: Transformers use self-attention mechanisms to process sequences in parallel, achieving better performance than RNNs on many NLP tasks.

### Q18: What is BERT?
**Answer**: BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model that can be fine-tuned for various NLP tasks.

**Code Example**:
```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input text
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([input_ids])

# Get embeddings
with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state

print(f"Input tokens: {tokens}")
print(f"Embedding shape: {embeddings.shape}")
```

## 10. Evaluation Metrics

### Q19: What are common evaluation metrics for NLP tasks?
**Answer**:
- Classification: Accuracy, Precision, Recall, F1-score
- Information Retrieval: Precision@K, Recall@K, MAP, NDCG
- Machine Translation: BLEU, ROUGE, METEOR
- Language Generation: Perplexity, BLEU, Human evaluation

**Code Example**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Example predictions and true labels
y_true = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")

print("\nDetailed Report:")
print(classification_report(y_true, y_pred))
```

## 11. Advanced Topics

### Q20: What is transfer learning in NLP?
**Answer**: Transfer learning involves using pre-trained models on large datasets and fine-tuning them for specific tasks, reducing training time and improving performance.

**Code Example**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load pre-trained model for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example prediction
text = "I love this movie!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    
print(f"Text: {text}")
print(f"Sentiment probabilities: {predictions}")
```

### Q21: What is the difference between extractive and abstractive summarization?
**Answer**:
- **Extractive**: Selects important sentences from the original text
- **Abstractive**: Generates new sentences that capture the main ideas

**Code Example**:
```python
from transformers import pipeline

# Extractive summarization using BERT
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language. NLP combines computational linguistics with statistical, machine 
learning, and deep learning models to enable computers to process and analyze 
large amounts of natural language data.
"""

summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
print("Summary:", summary[0]['summary_text'])
```

## 12. Coding Interview Questions

### Q22: Implement a function to find the most frequent word in a text
```python
def most_frequent_word(text):
    """Find the most frequent word in a text"""
    words = text.lower().split()
    word_count = {}
    
    for word in words:
        word = word.strip('.,!?";')  # Remove punctuation
        word_count[word] = word_count.get(word, 0) + 1
    
    return max(word_count, key=word_count.get)

# Test
text = "The quick brown fox jumps over the lazy dog. The dog was lazy."
print(f"Most frequent word: {most_frequent_word(text)}")
```

### Q23: Implement Jaccard similarity between two texts
```python
def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union != 0 else 0

# Test
text1 = "the quick brown fox"
text2 = "the lazy brown dog"
similarity = jaccard_similarity(text1, text2)
print(f"Jaccard similarity: {similarity:.3f}")
```

### Q24: Implement a simple spell checker
```python
def edit_distance(s1, s2):
    """Calculate edit distance between two strings"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def spell_check(word, dictionary, max_distance=2):
    """Simple spell checker using edit distance"""
    suggestions = []
    for dict_word in dictionary:
        if edit_distance(word, dict_word) <= max_distance:
            suggestions.append(dict_word)
    return suggestions

# Test
dictionary = ["hello", "world", "python", "programming", "language"]
word = "progaming"
suggestions = spell_check(word, dictionary)
print(f"Suggestions for '{word}': {suggestions}")
```

### Q25: Implement text classification using Naive Bayes
```python
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.vocabulary = set()
    
    def train(self, texts, labels):
        """Train the Naive Bayes classifier"""
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            words = text.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.vocabulary.add(word)
    
    def predict(self, text):
        """Predict the class of a text"""
        words = text.lower().split()
        class_scores = {}
        
        for class_label in self.class_counts:
            # Calculate log probability
            score = math.log(self.class_counts[class_label])
            
            for word in words:
                word_count = self.word_counts[class_label][word]
                total_words = sum(self.word_counts[class_label].values())
                vocab_size = len(self.vocabulary)
                
                # Laplace smoothing
                prob = (word_count + 1) / (total_words + vocab_size)
                score += math.log(prob)
            
            class_scores[class_label] = score
        
        return max(class_scores, key=class_scores.get)

# Test
texts = [
    "I love this movie",
    "This movie is terrible",
    "Great film, highly recommend",
    "Worst movie ever",
    "Amazing story and acting"
]
labels = ["positive", "negative", "positive", "negative", "positive"]

classifier = NaiveBayesClassifier()
classifier.train(texts, labels)

test_text = "This is a great movie"
prediction = classifier.predict(test_text)
print(f"Prediction for '{test_text}': {prediction}")
```

## Tips for NLP Interviews
- **Understand the basics**: Make sure you know tokenization, stemming, lemmatization, and basic preprocessing
- **Practice coding**: Be comfortable implementing basic NLP algorithms from scratch
- **Know your libraries**: Be familiar with NLTK, spaCy, scikit-learn, and transformers
- **Understand evaluation**: Know how to evaluate different types of NLP tasks
- **Stay updated**: Keep up with recent developments in transformers and large language models
- **Practice with real data**: Work on projects with messy, real-world text data
