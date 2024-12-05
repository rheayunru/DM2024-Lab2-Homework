import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load tweets_DM.json
with open('data/tweets_DM.json', "r") as file:
    tweets_data = [json.loads(line) for line in file]

tweets_df = pd.DataFrame([tweet["_source"]["tweet"] for tweet in tweets_data])

# Load emotion.csv
emotion_df = pd.read_csv('data/emotion.csv')

# Load data_identification.csv
data_id_df = pd.read_csv('data/data_identification.csv')
# Merge tweets with data identification
merged_df = pd.merge(tweets_df, data_id_df, on="tweet_id", how="inner")

# Merge with emotion labels (only training set will have labels)
merged_df = pd.merge(merged_df, emotion_df, on="tweet_id", how="left")
# Split data into training and testing
train_df = merged_df[merged_df["identification"] == "train"]
test_df = merged_df[merged_df["identification"] == "test"]
# Training data should have labels
X_train = train_df["text"]
y_train = train_df["emotion"]

# Testing data does not have labels
X_test = test_df["text"]


# Vectorize the text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# Transform training and testing data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict emotions for the test set
y_test_pred = model.predict(X_test_tfidf)

# Add predictions to the test DataFrame
test_df["predicted_emotion"] = y_test_pred
# Save predictions to a CSV file
test_df[["tweet_id", "predicted_emotion"]].to_csv("predicted_emotions.csv", index=False)
