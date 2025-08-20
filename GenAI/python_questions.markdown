# Simple Python Questions and Solutions

## Basic Python Questions (List, String, Dictionary, Set)

### 1. Reverse a String
Write a function to reverse a given string.

```python
def reverse_string(s: str) -> str:
    return s[::-1]

print(reverse_string("Fractal"))  # Output: "latcarF"
```

### 2. Check Palindrome
Write a function to check if a string is a palindrome (reads the same forward and backward).

```python
def is_palindrome(s: str) -> bool:
    return s == s[::-1]

print(is_palindrome("madam"))  # Output: True
print(is_palindrome("data"))   # Output: False
```

### 3. Find Second Highest Number in List
Write a function to find the second highest number in a list of integers.

```python
def second_highest(nums):
    first = second = float('-inf')
    for n in nums:
        if n > first:
            second, first = first, n
        elif first > n > second:
            second = n
    return None if second == float('-inf') else second

print(second_highest([2, 5, 1, 7, 7]))  # Output: 5
```

### 4. Find Missing Number in a Sequence
Given a list of n-1 integers in the range from 1 to n with no duplicates, find the missing number.

```python
def missing_number(nums):
    n = len(nums) + 1
    expected = n * (n+1) // 2
    return expected - sum(nums)

print(missing_number([1,2,3,5]))  # Output: 4
```

### 5. Sliding Window – Maximum Sum Subarray (Fixed Size k)
Find the maximum sum of any contiguous subarray of size k in a given array.

```python
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum

print(max_sum_subarray([2,1,5,1,3,2], 3))  # Output: 9 (5+1+3)
```

### 6. Two Sum Problem
Given an array of integers and a target sum, find two numbers that add up to the target. Return their indices.

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return (seen[target - num], i)
        seen[num] = i
    return None

print(two_sum([2,7,11,15], 9))  # Output: (0,1)
```

### 7. Count Word Frequency in a Text
Write a function to count the frequency of words in a given text.

```python
from collections import Counter

def word_count(text):
    words = text.lower().split()
    return Counter(words)

print(word_count("NLP is fun and NLP is powerful"))
# Output: {'nlp': 2, 'is': 2, 'fun': 1, 'and': 1, 'powerful': 1}
```

### 8. Longest Substring Without Repeating Characters
Find the length of the longest substring without repeating characters.

```python
def longest_unique_substring(s):
    seen, start, max_len = {}, 0, 0
    for i, ch in enumerate(s):
        if ch in seen and seen[ch] >= start:
            start = seen[ch] + 1
        seen[ch] = i
        max_len = max(max_len, i - start + 1)
    return max_len

print(longest_unique_substring("abcabcbb"))  # Output: 3
```

### 9. Prefix Sum (for Range Queries)
Compute the prefix sum of an array to efficiently calculate the sum of elements in a range.

```python
def prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(1, len(arr) + 1):
        prefix[i] = prefix[i-1] + arr[i-1]
    return prefix

arr = [1,2,3,4]
p = prefix_sum(arr)
# Query sum from index 1 to 3
print(p[3] - p[1])  # Output: 5 (2+3)
```

### 10. Merge Two Sorted Lists
Merge two sorted lists into a single sorted list.

```python
def merge_sorted_lists(a, b):
    i = j = 0
    merged = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    merged.extend(a[i:])
    merged.extend(b[j:])
    return merged

print(merge_sorted_lists([1,3,5], [2,4,6]))  # Output: [1,2,3,4,5,6]
```

### 11. Remove Duplicates from List
Write a function to remove duplicates from a list while preserving the order.

```python
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

print(remove_duplicates([1, 2, 2, 3, 4, 4, 5]))  # Output: [1, 2, 3, 4, 5]
```

### 12. Find Common Elements in Two Lists
Write a function to find common elements between two lists.

```python
def common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.intersection(set2))

print(common_elements([1, 2, 3, 4], [2, 4, 6, 8]))  # Output: [2, 4]
```

### 13. Count Characters in String
Write a function to count the frequency of each character in a string.

```python
def char_count(s: str) -> dict:
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

print(char_count("hello"))  # Output: {'h': 1, 'e': 1, 'l': 2, 'o': 1}
```

### 14. Find Unique Elements in List
Write a function to find elements that appear exactly once in a list.

```python
def unique_elements(lst):
    from collections import Counter
    counts = Counter(lst)
    return [item for item, count in counts.items() if count == 1]

print(unique_elements([1, 1, 2, 3, 3, 4]))  # Output: [2, 4]
```

### 15. Convert String to Dictionary
Write a function to convert a string of key-value pairs (e.g., "a=1,b=2,c=3") into a dictionary.

```python
def string_to_dict(s: str) -> dict:
    result = {}
    pairs = s.split(',')
    for pair in pairs:
        key, value = pair.split('=')
        result[key] = int(value)
    return result

print(string_to_dict("a=1,b=2,c=3"))  # Output: {'a': 1, 'b': 2, 'c': 3}
```

## Applied ML/NLP Coding Questions

### 16. TF-IDF Vectorization & Cosine Similarity
Calculate the TF-IDF vectors for a list of documents and compute the cosine similarity between two documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = ["I love NLP", "NLP is fun", "I enjoy machine learning"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

sim = cosine_similarity(X[0], X[1])
print(sim[0][0])  # Output: similarity score (e.g., 0.447...)
```

### 17. Simple Sentiment Classification (Logistic Regression)
Train a logistic regression model for sentiment classification using text data.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

texts = ["I love this product", "This is bad", "Amazing experience", "Terrible quality"]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

print(model.predict(vectorizer.transform(["I hate it"])))  # Output: [0]
```

### 18. Token Classification (NER-like)
Use spaCy to perform named entity recognition (NER) on a given text.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:
    print(ent.text, ent.label_)
# Output: Apple ORG, U.K. GPE, $1 billion MONEY
```

### 19. Word Embedding with Transformers
Generate word embeddings for sentences using a transformer model.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["NLP is powerful", "Deep learning is used in NLP"])
print(embeddings.shape)  # Output: (2, 384)
```

### 20. Detect Top Topics in Support Tickets
Use NMF to identify top topics in a list of support tickets.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

tickets = [
    "Password reset issue",
    "Login not working",
    "Server downtime",
    "Forgot password link not working",
    "Website is down"
]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(tickets)

nmf = NMF(n_components=2, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_

for i, topic in enumerate(H):
    top_words = [vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-3:]]
    print(f"Topic {i}: {top_words}")
# Output: Topic 0: ['password', 'reset', 'issue'], Topic 1: ['server', 'downtime', 'website']
```