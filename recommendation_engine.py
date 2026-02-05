import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class MovieRecommender:
    """Movie recommendation system using TF-IDF and Cosine Similarity"""
    
    def __init__(self):
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self, csv_file='imdb_movies_2024.csv'):
        """Load processed movie data"""
        print("Loading processed data...")
        self.df = pd.read_csv(csv_file)
        print(f"[OK] Loaded {len(self.df)} movies")
        
    def create_tfidf_matrix(self):
        """Create TF-IDF matrix from storylines"""
        print("\nCreating TF-IDF matrix...")
        
        # Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Fit and transform storylines
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.df['Processed_Storyline']
        )
        
        print(f"[OK] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
    def find_similar_movies(self, movie_name, top_n=5):
        """Find top N similar movies based on movie name"""
        
        # Find movie index
        movie_indices = self.df[self.df['Movie_Name'].str.contains(movie_name, case=False, na=False)].index
        
        if len(movie_indices) == 0:
            return None
        
        movie_idx = movie_indices[0]
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(
            self.tfidf_matrix[movie_idx:movie_idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get top N similar movies (excluding the movie itself)
        similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
        
        recommendations = []
        for idx in similar_indices:
            if idx != movie_idx:
                recommendations.append({
                    'Movie_Name': self.df.iloc[idx]['Movie_Name'],
                    'Storyline': self.df.iloc[idx]['Storyline'],
                    'Similarity_Score': round(cosine_similarities[idx] * 100, 2)
                })
        
        return recommendations[:top_n]
    
    def find_similar_by_storyline(self, user_storyline, top_n=5):
        """Find similar movies based on user input storyline"""
        
        # Preprocess user input
        from preprocess_data import clean_text
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            from nltk.stem import WordNetLemmatizer
            nltk_available = True
        except ImportError:
            nltk_available = False

        cleaned = clean_text(user_storyline)
        
        if nltk_available:
            try:
                stop_words = set(stopwords.words('english'))
                lemmatizer = WordNetLemmatizer()
                tokens = word_tokenize(cleaned)
                processed = ' '.join([lemmatizer.lemmatize(word) for word in tokens 
                                     if word not in stop_words and len(word) > 2])
            except LookupError:
                nltk_available = False
        
        if not nltk_available:
             processed = ' '.join([w for w in cleaned.split() if len(w) > 2])
        
        # Transform user storyline to TF-IDF
        user_tfidf = self.tfidf_vectorizer.transform([processed])
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        
        # Get top N similar movies
        similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'Movie_Name': self.df.iloc[idx]['Movie_Name'],
                'Storyline': self.df.iloc[idx]['Storyline'],
                'Similarity_Score': round(cosine_similarities[idx] * 100, 2)
            })
        
        return recommendations
    
    def save_model(self, filename='movie_recommender.pkl'):
        """Save the trained model"""
        model_data = {
            'df': self.df,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n[OK] Model saved to: {filename}")
    
    def load_model(self, filename='movie_recommender.pkl'):
        """Load a saved model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.df = model_data['df']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        
        print(f"[OK] Model loaded from: {filename}")

def train_model():
    """Train and save the recommendation model"""
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.create_tfidf_matrix()
    recommender.save_model()
    
    # Test the model
    print("\n[TEST] Testing model...")
    test_storyline = "A hero fights against evil forces to save the world"
    recommendations = recommender.find_similar_by_storyline(test_storyline, top_n=5)
    
    print(f"\nTest input: '{test_storyline}'")
    print("\nTop 5 recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['Movie_Name']} (Score: {rec['Similarity_Score']}%)")
    
    return recommender

if __name__ == "__main__":
    train_model()