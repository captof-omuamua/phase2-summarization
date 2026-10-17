#!/usr/bin/env python3
"""
preprocess.py

A script to clean and preprocess the dataset for Malayalam summarization.
It performs the following steps:
  1. Reads the CSV file (with columns "para" and "headline") from a hardcoded path.
  2. Removes duplicates and rows with missing values.
  3. Cleans the text in both "para" and "headline" by:
      - Normalizing whitespace.
      - Removing URLs.
      - Keeping only Malayalam characters, common punctuation, and whitespace.
      - Reducing repeated punctuation.
  4. If the punctuation ratio in a field (paragraph or headline) exceeds a set threshold,
     remove all punctuation from that field (instead of discarding the row).
  5. Saves the cleaned dataset as "news_90k_clean.csv" in the working directory.
  6. Prints a size analysis (number of rows and file size) along with counts of how many
     rows had punctuation removed.

Usage:
    python preprocess.py
"""

import re
import os
import pandas as pd

# Hardcoded file paths
INPUT_FILE = "news_90k.csv"
OUTPUT_FILE = "news_90k_clean.csv"  # saved in the working directory

# Parameters for cleaning
MIN_PARA_WORDS = 5
MAX_PARA_WORDS = 300
MIN_HEADLINE_WORDS = 3
MAX_HEADLINE_WORDS = 20
# If punctuation count per (word+1) exceeds this threshold, punctuation will be removed.
MAX_PUNCT_RATIO = 1.0  

def clean_text(text: str) -> str:
    """
    Clean input text by normalizing whitespace, removing URLs,
    and retaining only Malayalam characters, common punctuation, and whitespace.
    Also, reduces repeated punctuation to a single instance.
    """
    if pd.isnull(text):
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Retain only Malayalam characters (unicode \u0D00-\u0D7F), common punctuation, and whitespace.
    text = re.sub(r'[^\u0D00-\u0D7F\s\.,;:"\'?!\-–—]', '', text)
    
    # Reduce repeated punctuation (e.g., "!!!" becomes "!")
    text = re.sub(r'([,.?!\-–—])\1+', r'\1', text)
    
    return text

def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from the text.
    Keeps Malayalam characters, digits, and whitespace.
    """
    return re.sub(r'[^\u0D00-\u0D7F0-9\s]', '', text)

def punctuation_ratio(text: str) -> float:
    """
    Calculate the ratio of punctuation characters to (word count + 1) in the text.
    """
    words = text.split()
    punct_count = len(re.findall(r'[^\w\s]', text))
    return punct_count / (len(words) + 1)

def preprocess(input_csv: str, output_csv: str):
    # Read dataset
    df = pd.read_csv(input_csv)
    print(f"Original dataset size: {len(df)} rows")
    
    # Remove duplicates and rows with missing values in 'para' or 'headline'
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["para", "headline"], inplace=True)
    print(f"After removing duplicates and missing values: {len(df)} rows")
    
    # Clean text in both columns
    df["para"] = df["para"].apply(clean_text)
    df["headline"] = df["headline"].apply(clean_text)
    
    # Counters for punctuation removal
    para_punct_removed = 0
    headline_punct_removed = 0
    
    # Process each row: if punctuation ratio is high, remove punctuation from that field.
    def process_row(row):
        nonlocal para_punct_removed, headline_punct_removed
        para = row["para"]
        headline = row["headline"]
        
        # Check punctuation ratio for paragraph
        if punctuation_ratio(para) > MAX_PUNCT_RATIO:
            row["para"] = remove_punctuation(para)
            para_punct_removed += 1
        # Check punctuation ratio for headline
        if punctuation_ratio(headline) > MAX_PUNCT_RATIO:
            row["headline"] = remove_punctuation(headline)
            headline_punct_removed += 1
        return row
    
    df = df.apply(process_row, axis=1)
    
    # Save the cleaned dataset in the working directory
    df.to_csv(output_csv, index=False)
    print(f"\nCleaned dataset saved to: {output_csv}")
    
    # File size analysis
    file_size_bytes = os.path.getsize(output_csv)
    file_size_kb = file_size_bytes / 1024
    print(f"Final cleaned file size: {file_size_bytes} bytes ({file_size_kb:.2f} KB)")
    print(f"Total rows: {len(df)}")
    print(f"Rows with punctuation removed in 'para': {para_punct_removed}")
    print(f"Rows with punctuation removed in 'headline': {headline_punct_removed}")

def main():
    preprocess(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()
