import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load the saved model and data
movies_dict = pickle.load(open('model.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# TMDb API settings (using the API key you provided)
TMDB_API_KEY = '8265bd1679663a7ea12ac168da84d2e8'
TMDB_SEARCH_URL = 'https://api.themoviedb.org/3/search/movie'

# Function to fetch movie ID based on the title
def fetch_movie_id(movie_title):
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_title
    }
    response = requests.get(TMDB_SEARCH_URL, params=params)
    data = response.json()
    if data['results']:
        return data['results'][0]['id']  # Return the first movie's ID
    return None

# Function to fetch poster using movie ID
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US".format(movie_id, TMDB_API_KEY)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Function to calculate similarity and recommend movies
def recommend(movie, num_recommendations=5):
    # Get the index of the movie from the dataframe
    index = movies[movies['title'] == movie].index[0]
    
    # Calculate cosine similarity
    distances = similarity_matrix[index]
    
    # Get the list of movie indices sorted by similarity
    movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
    
    # Fetch movie titles and posters
    recommended_movie_names = []
    recommended_movie_posters = []
    
    for i in movie_indices:
        movie_title = movies.iloc[i[0]].title
        movie_id = fetch_movie_id(movie_title)
        poster_url = fetch_poster(movie_id) if movie_id else None
        recommended_movie_names.append(movie_title)
        recommended_movie_posters.append(poster_url)
    
    return recommended_movie_names, recommended_movie_posters

# Load the similarity matrix from a precomputed model
similarity_matrix = cosine_similarity(pickle.load(open('vector.pkl', 'rb')))

# Streamlit App UI
st.title('Movie Recommendation System')

# Dropdown box for movie selection
movie_list = movies['title'].values  # Movie titles from the dataframe
selected_movie = st.selectbox('Select a movie to get recommendations', movie_list)

# Slider to select the number of recommendations
num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

# If a movie is selected and the button is clicked, display recommendations
if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie, num_recommendations)
    
    # Display recommendations in rows with a maximum of 3 columns per row
    for i in range(0, num_recommendations, 3):
        cols = st.columns(3)  # Create up to 3 columns for each row
        for j in range(3):
            if i + j < num_recommendations:
                with cols[j]:
                    st.text(recommended_movie_names[i + j])  # Display movie name
                    st.image(recommended_movie_posters[i + j], use_column_width=True)  # Display movie poster
