import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load datasets
# Assuming df_train has 'comment' and 'sentiment', and df_implementation is your second dataset
df_train = pd.read_csv('data/train.csv')
df_implementation = pd.read_csv('data/test.csv')

# Train the Naive Bayes model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df_train['comment'])
y_train = df_train['sentiment']
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict sentiment and confidence for implementation dataset
X_implementation = vectorizer.transform(df_implementation['comment'])
df_implementation['predicted_sentiment'] = model.predict(X_implementation)

# Get the confidence (probability of the predicted class)
proba_implementation = model.predict_proba(X_implementation)
confidence_scores = proba_implementation.max(axis=1)  # Confidence of the predicted class
df_implementation['confidence'] = confidence_scores * 100  # Convert to percentage

# Display the table with comment, predicted sentiment, and confidence
st.title("Sentiment Prediction System")
st.subheader("Dataset Predictions")
st.dataframe(df_implementation[['comment', 'predicted_sentiment', 'confidence']])

# User input section for new comment prediction
st.subheader("Input your own comment for sentiment prediction")
user_comment = st.text_input("Enter your comment here:")

if user_comment:
    user_input_vector = vectorizer.transform([user_comment])
    user_pred = model.predict(user_input_vector)[0]
    
    # Get confidence for the user's input prediction
    user_proba = model.predict_proba(user_input_vector)
    user_confidence = user_proba.max() * 100  # Confidence of the predicted class in percentage
    
    st.write(f"Predicted Sentiment: {user_pred}")
    st.write(f"Confidence: {user_confidence:.2f}%")
