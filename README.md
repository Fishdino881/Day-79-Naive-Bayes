# Day-79-Naive-Bayes

###  Overview

**Naive Bayes** is a **supervised machine learning algorithm** based on **Bayes’ Theorem**.
It is mainly used for **classification problems**, especially with text data.

---

##  How Naive Bayes Works

It calculates the probability of a class given the input features.

Formula:

```id="p1k9sd"
P(A|B) = (P(B|A) * P(A)) / P(B)
```

Where:

* **P(A|B)** → Posterior probability
* **P(B|A)** → Likelihood
* **P(A)** → Prior
* **P(B)** → Evidence

---

##  Key Assumption

Naive Bayes assumes that **features are independent** of each other.

---

##  Types of Naive Bayes

* **Gaussian Naive Bayes** → Continuous data
* **Multinomial Naive Bayes** → Text data (word counts)
* **Bernoulli Naive Bayes** → Binary features

---

##  Implementation (Python)

```python id="k8z4dj"
from sklearn.naive_bayes import GaussianNB
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

model = GaussianNB()
model.fit(X, y)

prediction = model.predict([[2.5]])
print(prediction)
```

---

##  Advantages

- Fast and efficient
- Works well with large datasets
- Performs well in text classification

---

##  Disadvantages

* Assumes feature independence (not always true)
* Less accurate with complex relationships

---

##  Use Cases

* Spam detection
* Sentiment analysis
* Document classification
* Recommendation systems

---

##  Key Takeaways

- Based on probability and Bayes’ theorem
- Simple and fast algorithm
- Strong performance in text-based tasks

---

