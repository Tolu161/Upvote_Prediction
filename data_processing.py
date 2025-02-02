#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:39:29 2025

@author: toluojo
"""

'''
DATA PROCESSING 
'''

# Fetching titles from PostgreSQL.
# Cleaning, tokenisation, and lemmatisation.
# Generating word_to_index and exporting it.

import psycopg2
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# Connection details (ensure these are kept secure)
host = "178.156.142.230"
database = "hd64m1ki"
user = "sy91dhb"
password = "g5t49ao"
port = "5432"

def fetch_data():
    """Fetches titles from PostgreSQL and returns them as a Pandas Series."""
    try:
        conn = psycopg2.connect(
            host=host, database=database, user=user, password=password, port=port
        )
        cursor = conn.cursor()
        query = """
            SELECT title FROM hacker_news.items 
            WHERE type = 'story' AND title IS NOT NULL 
            LIMIT 100000;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return pd.Series([row[0] for row in rows])  # Convert to Pandas Series
    except Exception as e:
        print(f"Database error: {e}")
        return pd.Series([])

def normalize_text(text):
    """Lowercase and clean text."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

def preprocess_text(tokens, lemmatizer, stop_words):
    """Lemmatise tokens and remove stopwords."""
    return [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]

# Fetch and process titles
titles = fetch_data()
normalized_titles = titles.apply(normalize_text)

# Tokenisation
tokenized_titles = [word_tokenize(title) for title in normalized_titles]

# Initialise NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Process text
processed_titles = [preprocess_text(tokens, lemmatizer, stop_words) for tokens in tokenized_titles]

# Create word-to-index dictionary
all_words = set(word for title in processed_titles for word in title)
word_to_index = {word: idx for idx, word in enumerate(all_words)}

# Export functions and dictionary
__all__ = ["word_to_index", "normalize_text", "preprocess_text"]
