# NM-HACKATHON-SET-2
AI-Powered Movie Recommendation System

# ---------------------------- app.py ----------------------------
import streamlit as st
import pandas as pd
import requests
import pickle

# Load movie data and similarity matrix
movies = pd.read_csv('movies.csv')
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Function to fetch movie poster using TMDB API
def fetch_poster(movie_id):
    api_key = "your_tmdb_api_key"  # 🔑 Replace with your TMDB API key
    response = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    )
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

# Recommendation logic
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_posters = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters

# Streamlit UI
st.title("🎬 AI-Powered Movie Recommendation System")
selected_movie = st.selectbox("Search your favorite movie", movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, col in enumerate([col1, col2, col3, col4, col5]):
        with col:
            st.text(names[i])
            st.image(posters[i])

# ----------------------- generate_similarity.py -----------------------

"""
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])
similarity = cosine_similarity(tfidf_matrix)

pickle.dump(similarity, open('similarity.pkl', 'wb'))
print("Similarity matrix saved to similarity.pkl")
"""

# --------------------------- movies.csv ---------------------------
"""
movie_id,title,tags
101,The Matrix,action sci-fi hacker neo bullet-time
102,Inception,dream heist mind-bending sci-fi thriller
103,Interstellar,space time relativity blackhole nasa sci-fi
104,The Dark Knight,batman joker hero crime thriller
105,Avengers,superhero marvel action team ironman
"""

# ------------------------ requirements.txt ------------------------
"""
streamlit
pandas
scikit-learn
requests
"""

# ----------------------------- README.md -----------------------------
"""
# 🎬 AI-Powered Movie Recommendation System

An intelligent movie recommendation system using content-based filtering, powered by TF-IDF & cosine similarity.

## ✅ Features
- Search for a movie and get 5 similar recommendations
- View movie posters fetched from TMDB
- Lightweight, fast, and user-friendly

## 🚀 Tech Stack
- Python
- Pandas, Scikit-learn
- Streamlit
- TMDB API

## 📦 Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
