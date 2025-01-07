import sys
import pandas as pd
import numpy as np
import re
import string
import math
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class SentimentAnalyzer:
    def __init__(self):
        # We'll need a comprehensive list of stop words to clean our text
        self.stop_words = self._get_stop_words()
            
        # These will store our model's learned probabilities
        self.class_priors = {}
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()
        
    def _get_stop_words(self):
        """
        Create a detailed list of stop words to remove noise from our text.
        This helps improve the quality of our text analysis.
        """
        return set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
            'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 
            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 
            'for', 'with', 'about', 'against', 'between', 'into', 
            'through', 'during', 'before', 'after', 'above', 'below', 
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 
            'such', 'no', 'nor', 'not', 's', 't', 'can', 'will', 
            'just', 'don', "don't", 'should', "should've", 'now', 'd', 
            'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
            'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
            'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
            'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
        ])
    
    def preprocess_text(self, text):
        """
        Clean and prepare text for analysis by removing noise and irrelevant information.
        This step is crucial for improving classification accuracy.
        """
        if pd.isna(text):
            return []
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize and remove stop words
        words = [
            word for word in text.split() 
            if word not in self.stop_words and len(word) > 1
        ]
        
        return words
    
    def fit(self, X, y):
        """
        Train the model by learning the probability distributions of words and classes.
        This is where the magic of machine learning happens!
        """
        # Remove invalid entries
        valid_indices = pd.notna(X)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Calculate class priors with Laplace smoothing
        total_docs = len(y)
        self.class_counts = pd.Series(y).value_counts()
        for class_label in self.class_counts.index:
            self.class_priors[class_label] = (self.class_counts[class_label] + 1) / (total_docs + len(self.class_counts))
        
        # Count word frequencies per class
        word_counts = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        
        for doc, label in zip(X, y):
            words = self.preprocess_text(doc)
            for word in words:
                self.vocabulary.add(word)
                word_counts[label][word] += 1
        
        # Calculate word probabilities with Laplace smoothing
        vocab_size = len(self.vocabulary)
        for class_label in self.class_counts.index:
            total_words = sum(word_counts[class_label].values()) + vocab_size
            for word in self.vocabulary:
                count = word_counts[class_label][word] + 1
                self.word_probs[class_label][word] = count / total_words
    
    def predict_prob(self, text):
        """
        Predict the probability of each class for a given text.
        We use log probabilities to prevent underflow and improve numerical stability.
        """
        words = self.preprocess_text(text)
        scores = {}
        
        for class_label in self.class_priors:
            score = math.log(self.class_priors[class_label])
            for word in words:
                if word in self.vocabulary:
                    score += math.log(self.word_probs[class_label][word])
            scores[class_label] = score
        
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: v/total for k, v in exp_scores.items()}
    
    def predict(self, X):
        """
        Predict the most likely class for input text(s).
        Handles both single text and multiple text inputs.
        """
        if isinstance(X, str):
            X = [X]
        return [max(self.predict_prob(text).items(), key=lambda x: x[1])[0] for text in X]

def calculate_metrics(y_true, y_pred):
    # (Metrics calculation remains the same as in the original code)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_true_binary = (y_true == 'Positive').astype(int)
    y_pred_binary = (y_pred == 'Positive').astype(int)
    
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'accuracy': accuracy,
        'f_score': f_score
    }

def main():
    # Default parameters if no command line arguments
    algo = 0
    train_size = 80

    # Process command line arguments
    if len(sys.argv) == 3:
        algo = int(sys.argv[1])
        train_size = int(sys.argv[2])
        if train_size < 50 or train_size > 80:
            train_size = 80

    try:
        # Read and preprocess data
        data = pd.read_csv('twitter_training.csv', names=['id', 'game', 'sentiment', 'text'])
        data = data.dropna(subset=['text', 'sentiment']).reset_index(drop=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'], 
            data['sentiment'], 
            train_size=train_size/100, 
            random_state=42,
            stratify=data['sentiment']
        )

    except Exception as e:
        print(f"Error reading or processing the dataset: {str(e)}")
        return

    # Choose classifier based on algorithm parameter
    if algo == 0:
        classifier = SentimentAnalyzer()
        classifier_type = "Naive Bayes"
    else:
        classifier = Pipeline([
            ('tfidf', TfidfVectorizer(
                preprocessor=lambda x: '' if pd.isna(x) else x,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('logistic', LogisticRegression(
                max_iter=2000, 
                class_weight='balanced',
                solver='liblinear'
            ))
        ])
        classifier_type = "Logistic Regression"

    # Print configuration
    print(f"Patel, Tithi, A20580634 solution:")
    print(f"\nTraining set size: {train_size} %")
    print(f"Classifier type: {classifier_type}")
    
    # Train classifier
    print("\nTraining classifier...")
    classifier.fit(X_train, y_train)
    
    # Test classifier
    print("\nTesting classifier...")
    y_pred = classifier.predict(X_test)
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    print("\nTest results / metrics:")
    print(f"Number of true positives: {metrics['true_positives']}")
    print(f"Number of true negatives: {metrics['true_negatives']}")
    print(f"Number of false positives: {metrics['false_positives']}")
    print(f"Number of false negatives: {metrics['false_negatives']}")
    print(f"Sensitivity (recall): {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Negative predictive value: {metrics['npv']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F-score: {metrics['f_score']:.4f}")

    # Interactive prediction loop
    while True:
        print("\nEnter your sentence/document:")
        sentence = input("Sentence/document S: ")
        
        if algo == 0:
            probs = classifier.predict_prob(sentence)
            prediction = max(probs.items(), key=lambda x: x[1])[0]
            print(f"\nwas classified as {prediction}.")
            for label in sorted(probs.keys()):
                print(f"P({label} | S) = {probs[label]:.4f}")
        else:
            prediction = classifier.predict([sentence])[0]
            if hasattr(classifier, 'predict_prob'):
                probabilities = classifier.predict_prob([sentence])[0]
                class_labels = classifier.named_steps['logistic'].classes_
                print(f"\nwas classified as {prediction}.")
                for label, prob in zip(class_labels, probabilities):
                    print(f"P({label} | S) = {prob:.4f}")
            else:
                print(f"\nwas classified as {prediction}.")
        
        cont = input("\nDo you want to enter another sentence [Y/N]? ")
        if cont.upper() != 'Y':
            break

if __name__ == "__main__":
    main()