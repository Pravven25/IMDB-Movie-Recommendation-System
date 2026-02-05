"""
Load IMDb Official Datasets
This is MUCH easier and more reliable than web scraping!
"""

import pandas as pd
import requests
import gzip
import shutil
from io import BytesIO

def download_and_filter_basics():
    """Download and filter basics dataset using chunks to save memory"""
    url = 'https://datasets.imdbws.com/title.basics.tsv.gz'
    print("Downloading and filtering basics (this may take a moment)...")
    
    chunks = []
    try:
        # Use chunksize to avoid Out Of Memory errors
        for chunk in pd.read_csv(url, sep='\t', low_memory=False, compression='gzip', chunksize=100000):
            # Filter immediately within the chunk
            filtered_chunk = chunk[
                (chunk['titleType'] == 'movie') & 
                (chunk['startYear'] == '2024')
            ]
            chunks.append(filtered_chunk)
        
        basics_df = pd.concat(chunks)
        print(f"✓ basics: Found {len(basics_df):,} movies from 2024")
        return basics_df
    except Exception as e:
        print(f"✗ Error processing basics: {e}")
        return None

def download_ratings():
    """Download ratings dataset"""
    url = 'https://datasets.imdbws.com/title.ratings.tsv.gz'
    print("Downloading ratings...")
    try:
        return pd.read_csv(url, sep='\t', low_memory=False, compression='gzip')
    except Exception as e:
        print(f"✗ Error downloading ratings: {e}")
        return None

def create_movie_dataset():
    """Create a clean movie dataset for the project"""
    
    print("="*70)
    print("🎬 IMDb 2024 Movie Dataset Creator (Memory Optimized)")
    print("="*70 + "\n")
    
    # Process basics in chunks
    basics_2024 = download_and_filter_basics()
    if basics_2024 is None:
        print("\n❌ Failed to process basics dataset")
        create_sample_fallback()
        return

    # Process ratings
    ratings = download_ratings()
    if ratings is None:
        print("\n❌ Failed to process ratings dataset")
        create_sample_fallback()
        return

    # Merge
    print("\n🔍 Merging datasets...")
    movies_with_ratings = basics_2024.merge(
        ratings[['tconst', 'averageRating', 'numVotes']],
        on='tconst',
        how='left'
    )
    
    # Sort by popularity
    movies_with_ratings['numVotes'] = pd.to_numeric(movies_with_ratings['numVotes'], errors='coerce').fillna(0)
    movies_with_ratings = movies_with_ratings.sort_values('numVotes', ascending=False)

    # Create simplified dataset
    movie_list = []
    print("📝 Creating movie dataset...")
    
    for idx, row in movies_with_ratings.iterrows():
        genres = str(row['genres']).replace('\\N', 'Unknown')
        title = row['primaryTitle']
        year = row['startYear']
        rating = row.get('averageRating', 'N/A')
        
        storyline = f"This {genres.lower()} feature film released in {year} stars as {title}. It has received a rating of {rating} on IMDb and is a notable contribution to the {genres.lower()} genre."
        
        movie_list.append({
            'Movie_Name': title,
            'Storyline': storyline,
            'Year': year,
            'Rating': rating,
            'Genres': genres
        })
    
    df = pd.DataFrame(movie_list)
    df.to_csv('imdb_movies_2024.csv', index=False, encoding='utf-8')
    
    print(f"\n✨ Dataset created successfully!")
    print(f"📊 Total movies: {len(df):,}")
    print(f"📂 Saved to: imdb_movies_2024.csv")

def create_sample_fallback():
    """Create sample data if download fails"""
    
    from create_sample_data import sample_movies
    import pandas as pd
    
    df = pd.DataFrame(sample_movies)
    df.to_csv('imdb_movies_2024.csv', index=False, encoding='utf-8')
    print(f" Created sample dataset with {len(df)} movies")

if __name__ == "__main__":
    create_movie_dataset()