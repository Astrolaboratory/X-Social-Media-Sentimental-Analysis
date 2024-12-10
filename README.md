# 👹 X Social Media Sentiment Analysis

![Python Badge](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)
![TensorFlow Badge](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn Badge](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy Badge](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)
![Numpy Badge](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas Badge](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly Badge](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)
![Keras Badge](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![Conda Badge](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)

## Problem Statement 📝

In the current age of digital communication, social media platforms such as Twitter are frequently used by individuals to share opinions, thoughts, and emotions. These short, often unstructured, texts are rich in insights but difficult to analyze manually. Sentiment analysis, a branch of Natural Language Processing (NLP), helps identify whether the sentiment conveyed in text is positive, negative, or neutral.

This project aims to develop a **Twitter Sentiment Analysis** system to analyze tweets and classify them into sentiments. Understanding sentiment can be particularly valuable for businesses to gauge customer sentiment around product launches, company reputation, or public reactions to events. Through sentiment analysis, companies can efficiently gather customer feedback and take data-driven actions.

---

## Machine Learning and Data Science 🤖

### Text Data Processing

The core challenge of text data analysis is converting unstructured textual information into structured data that can be fed to machine learning models. For this, we use techniques such as **vectorization** and **word embeddings**.

- **Vectorization** converts the text data into numerical form. Words are represented as vectors, which allows machine learning algorithms to process them.
- **Word Embeddings**, like **Word2Vec** or **GloVe**, represent words as dense vectors, capturing semantic meaning in a higher-dimensional space.

### Sentiment Classification with Machine Learning

To classify sentiments (positive, negative, or neutral), we use machine learning algorithms, specifically deep learning models such as **Deep Neural Networks (DNN)**. These models are ideal for handling the complexities of language and can capture patterns in large datasets.

We employ **Supervised Learning**, where labeled datasets (tweets with known sentiment labels) are used to train the model. Once trained, the model can predict the sentiment of new, unseen tweets.

---

## Natural Language Processing (NLP) 🧠

NLP is a crucial component in this project as it allows us to understand and interpret human language in a way that machines can process.

### Steps Involved in NLP:

1. **Text Preprocessing**: This includes steps like tokenization (splitting text into words or phrases), lowercasing, removing stopwords (common words like 'and', 'the', etc.), and lemmatization (reducing words to their root form).
   
2. **Vectorization**: After preprocessing, we convert text into a numerical form using **Count Vectorizer** or **TF-IDF Vectorizer**. These methods create a matrix of token counts or weighted word frequencies, which can then be used by the machine learning model.
   - **Count Vectorizer** counts the frequency of each word in the text.
   - **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency) takes into account the importance of a word in the document, giving more weight to terms that are rare across the corpus but frequent within a document.

3. **Model Training**: After vectorizing the text, it is fed into a machine learning model. For this project, we use **Deep Neural Networks (DNNs)**, which are known for their high capacity to learn complex patterns.

---

## Vectorizers 🧑‍💻

To turn raw text data into a form that can be processed by machine learning algorithms, we use vectorization techniques.

- **Count Vectorizer**: This method counts the occurrence of each word in the text. Each word becomes a feature in a vector, where the value represents the word's frequency. The key limitation of this method is that it doesn’t account for the importance of words across different documents.
  
- **TF-IDF Vectorizer**: Unlike Count Vectorizer, the TF-IDF method assigns a weight to each word based on its importance in the document compared to the entire corpus. This helps in reducing the weight of common words (like "is", "the") and increases the weight of rare but significant words (like "fantastic", "disappointment").

---

## Machine Learning Models 🤖

We use **Deep Neural Networks (DNNs)** for sentiment classification. This model is chosen because it can learn complex non-linear relationships within the text data and adapt as more data is fed into the model.

- **Model Architecture**: A typical DNN includes multiple layers (input, hidden, and output layers). In this project, the input layer processes the vectorized text, and the output layer generates predictions about the sentiment of the tweet.
  
- **Activation Function**: **ReLU (Rectified Linear Unit)** is used as the activation function for the hidden layers, which helps the model learn complex relationships by introducing non-linearity. The output layer uses the **Softmax** activation function, which converts the model's outputs into probabilities for each class (positive, negative, or neutral sentiment).

---

## Exploratory Data Analysis (EDA) 📊

Before diving into machine learning, we perform **Exploratory Data Analysis (EDA)** to better understand the dataset and identify any patterns that can guide feature engineering.

### Key Insights from EDA:
1. **Sentiment Distribution**: A **countplot** of the sentiment labels (positive, negative, and neutral) revealed that neutral tweets were more frequent, while positive and negative tweets were less prevalent.
2. **Word Clouds**: Visualizing the most frequent words in positive and negative tweets using **word clouds** helped us understand the general mood expressed in each category. For example:
   - **Positive Sentiments**: Words like "good," "awesome," "happy," and "great" were more frequent.
   - **Negative Sentiments**: Words such as "hate," "sorry," and "disappointing" appeared more frequently.

The word clouds helped confirm that the vocabulary associated with positive and negative tweets was distinct, which can be crucial for training the model.

---

## Hyperparameter Tuning 🛠️

Hyperparameter tuning is an essential step to optimize model performance. In this project, we tested different hyperparameters such as:
- **Number of hidden layers** in the neural network.
- **Learning rate**: Controls the speed at which the model updates its parameters.
- **Batch size**: The number of training samples used in one iteration.
- **Epochs**: The number of times the model is trained on the entire dataset.

We used techniques like **Grid Search** or **Random Search** to find the best combination of parameters that yielded the highest accuracy.

---

## Results 📈

### Training and Test Loss

After training the model, we observed that the training loss was significantly lower than the test loss, indicating **overfitting**. Overfitting occurs when a model learns the training data too well, capturing noise instead of general patterns. Although the model performed well on training data, it had reduced accuracy on the test data. This suggests the need for techniques like **regularization** or **cross-validation** to improve generalization.

![Training and Test Loss](https://github.com/user-attachments/assets/ebc1260c-3cec-43c3-9666-840044e215f0)

---

## Conclusion 🔚

This project demonstrates how to apply machine learning and Natural Language Processing techniques to analyze and classify the sentiment of tweets. By utilizing deep learning models, we were able to predict whether the sentiment in a tweet was positive, negative, or neutral, based on the words and phrases used.

The ability to analyze Twitter sentiment provides significant value to businesses, enabling them to quickly understand customer reactions and adjust their strategies accordingly. In future iterations of this project, we can further improve model accuracy by:
- Collecting a more diverse dataset.
- Using advanced neural network architectures like **LSTM** (Long Short-Term Memory) or **BERT** (Bidirectional Encoder Representations from Transformers).
- Experimenting with more sophisticated hyperparameter tuning techniques.

This sentiment analysis model can be further enhanced by integrating additional features such as tweet metadata (e.g., user location, time of day) to increase the robustness of predictions.

---

## References 📚
1. [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
2. [Natural Language Processing with Python](https://www.oreilly.com/library/view/natural-language-processing/9780596803346/)
3. [TensorFlow Documentation](https://www.tensorflow.org/)

