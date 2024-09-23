import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.title("Sentiment Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("hf://datasets/lllaurenceee/Shopee_Bicycle_Reviews/labeled_comments.csv")

df = load_data()

# Function to handle sentiment analysis for the uploaded dataset
def sentiment_analysis(df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Comment'])
    y = df['sentiment']

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Naive Bayes Model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predictions and accuracies
    df['Predicted_Sentiment'] = model.predict(X)
    df['Sentiment_Accuracy'] = model.predict_proba(X).max(axis=1) * 100  # Confidence percentage
    df['Sentiment_Accuracy'] = df['Sentiment_Accuracy'].round(2).astype(str) + '%'  # Formatting as percentage

    return df[['Comment', 'Predicted_Sentiment', 'Sentiment_Accuracy']], vectorizer, model

# Function to display output
def display_output(df):
    st.write(df)

# Function to predict sentiment for user input
def predict_sentiment(user_comment, vectorizer, model):
    user_comment_transformed = vectorizer.transform([user_comment])
    predicted_sentiment = model.predict(user_comment_transformed)[0]
    sentiment_accuracy = model.predict_proba(user_comment_transformed).max(axis=1)[0] * 100  # Confidence percentage
    return predicted_sentiment, sentiment_accuracy
    
# Perform sentiment analysis
result_df, vectorizer, model = sentiment_analysis(df)
    
# Display the analysis result
display_output(result_df)
    
# User input for live sentiment prediction
st.markdown("### Predict Sentiment for Your Own Comment")
user_comment = st.text_input("Enter a comment:")
    
if user_comment:
    predicted_sentiment, accuracy = predict_sentiment(user_comment, vectorizer, model)
    st.write(f"**Predicted Sentiment:** {predicted_sentiment}")
    st.write(f"**Accuracy:** {accuracy:.2f}%")
