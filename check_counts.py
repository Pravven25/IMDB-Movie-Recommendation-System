import pandas as pd
import os

files = ['imdb_movies_2024.csv', 'imdb_movies_processed.csv']
for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f"{f}: {len(df)}")
    else:
        print(f"{f}: Not found")
