import pandas as pd
import re
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import nltk
    # Ensure NLTK data is downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available, using simple text processing.")

def clean_text(text):
    """Clean and preprocess text data"""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_storylines(csv_file='imdb_movies_2024.csv'):
    """Preprocess movie storylines using NLP techniques"""
    
    print("Loading data...")
    df = pd.read_csv(csv_file)
    
    print(f"Total movies: {len(df)}")
    
    # Remove movies without storylines
    df = df[df['Storyline'].notna()]
    df = df[df['Storyline'] != 'No storyline available']
    df = df[df['Storyline'].str.len() > 20]  # At least 20 characters
    
    print(f"Movies with valid storylines: {len(df)}")
    
    # Initialize NLP tools
    use_nltk = NLTK_AVAILABLE
    if use_nltk:
        try:
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
        except LookupError:
            print("Warning: NLTK data missing, using simple text processing.")
            use_nltk = False

    
    
    # Clean storylines
    print("\nPreprocessing storylines...")
    df['Cleaned_Storyline'] = df['Storyline'].apply(clean_text)
    
    # Tokenize and remove stopwords
    def process_text(text):
        if use_nltk:
            try:
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(word) for word in tokens 
                         if word not in stop_words and len(word) > 2]
                return ' '.join(tokens)
            except Exception:
                pass # Fallback
        
        # Simple fallback
        return ' '.join([w for w in text.split() if len(w) > 2])
    
    df['Processed_Storyline'] = df['Cleaned_Storyline'].apply(process_text)
    
    # Save processed data
    df.to_csv('imdb_movies_processed.csv', index=False, encoding='utf-8')
    print(f"\n Preprocessing complete!")
    print(f"Saved to: imdb_movies_processed.csv")
    
    # Show sample
    print("\n[OK] Sample processed data:")
    print(df[['Movie_Name', 'Processed_Storyline']].head(3))
    
    return df

if __name__ == "__main__":
    preprocess_storylines()