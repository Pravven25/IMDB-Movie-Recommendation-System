# IMDB Movie Recommendation System Using Storylines

## Overview
This project recommends movies based on storyline similarity using Natural Language Processing (NLP).

## Tech Stack
- Python
- Selenium (Web Scraping)
- Pandas
- Scikit-learn (TF-IDF, Cosine Similarity)
- Streamlit
- NLTK / SpaCy

## How It Works
1. Scrape IMDb 2024 movies using Selenium
2. Clean and preprocess storylines
3. Convert text into vectors using TF-IDF
4. Compute similarity using Cosine Similarity
5. Recommend Top 5 similar movies

## How to Run
```bash
pip install -r requirements.txt
python imdb_scraper.py
streamlit run app.py

