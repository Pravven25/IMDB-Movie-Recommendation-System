import pickle
import os

model_path = 'movie_recommender.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        df = data.get('df')
        print(f"Movies in {model_path}: {len(df)}")
else:
    print(f"{model_path} not found")
