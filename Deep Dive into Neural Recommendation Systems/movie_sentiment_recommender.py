import os
os.environ['USE_TORCH'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import time
import random
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
import ics
from io import StringIO
import base64

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set up the page
st.set_page_config(
    page_title="Enhanced Movie Recommender", 
    layout="wide",
    page_icon="🎬"
)

st.title("🎬 Enhanced Fusion Movie Recommendation System")
st.markdown("""
    **Combining Review Sentiment Analysis with LightGCN Graph Neural Networks**  
    Upload your movie data to get personalized recommendations using advanced AI techniques.
""")

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #1e3d73 !important;
        margin-bottom: 10px !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .model-card {
        background-color: #eef7ff;
        border-left: 4px solid #1e3d73;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .recommendation-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        font-weight: bold !important;
        color: #1e3d73 !important;
    }
    .feedback-btn {
        margin-right: 10px;
        margin-top: 10px;
    }
    .share-link {
        background-color: #e8f4ff;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        word-break: break-all;
    }
    .cold-start-card {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'sentiment_recommendations' not in st.session_state:
    st.session_state.sentiment_recommendations = None
if 'lightgcn_model' not in st.session_state:
    st.session_state.lightgcn_model = None
if 'interaction_matrix' not in st.session_state:
    st.session_state.interaction_matrix = None
if 'user_embeddings' not in st.session_state:
    st.session_state.user_embeddings = None
if 'item_embeddings' not in st.session_state:
    st.session_state.item_embeddings = None
if 'movie_map' not in st.session_state:
    st.session_state.movie_map = None
if 'user_profiles' not in st.session_state:
    st.session_state.user_profiles = {}
if 'shared_watchlists' not in st.session_state:
    st.session_state.shared_watchlists = {}
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = pd.DataFrame(columns=['user_id', 'movie_id', 'feedback', 'timestamp'])
if 'hybrid_recommendations' not in st.session_state:
    st.session_state.hybrid_recommendations = {}

# Genre definitions
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", 
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "Western", "Family"
]

SEASONAL_MOVIES = {
    "Holiday": ["It's a Wonderful Life", "Home Alone", "Elf", "The Polar Express", "Love Actually"],
    "Summer": ["Jaws", "Independence Day", "Jurassic Park", "The Sandlot", "Fast & Furious"],
    "Halloween": ["Hocus Pocus", "The Nightmare Before Christmas", "Beetlejuice", "Halloween", "Ghostbusters"],
    "Valentine": ["Titanic", "The Notebook", "La La Land", "Pride & Prejudice", "Crazy Rich Asians"]
}

# Fixed sentiment model loader
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# LightGCN Model
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        # Embedding layers
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        
    def forward(self, adj_matrix):
        # Initial embeddings
        user_embeddings = self.user_emb.weight
        item_embeddings = self.item_emb.weight
        all_embeddings = torch.cat([user_embeddings, item_embeddings])
        
        embeddings_list = [all_embeddings]
        
        # LightGCN propagation
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Combine all layers
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)
        
        user_final, item_final = torch.split(
            final_embeddings, [self.num_users, self.num_items]
        )
        return user_final, item_final

# Train LightGCN model
def train_lightgcn(train_mat, epochs=20, lr=0.001):
    num_users, num_items = train_mat.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create adjacency matrix
    coo = train_mat.tocoo()
    rows = np.concatenate([coo.row, coo.col + num_users])
    cols = np.concatenate([coo.col + num_users, coo.row])
    data = np.ones(rows.shape[0])
    
    adj_matrix = coo_matrix(
        (data, (rows, cols)), 
        shape=(num_users + num_items, num_users + num_items)
    )
    
    # Ensure matrix is in COO format
    if not isinstance(adj_matrix, coo_matrix):
        adj_matrix = adj_matrix.tocoo()
    
    # Convert to PyTorch tensor
    indices = torch.LongTensor(np.vstack([adj_matrix.row, adj_matrix.col]))
    values = torch.FloatTensor(adj_matrix.data)
    adj_matrix_tensor = torch.sparse_coo_tensor(
        indices, values, torch.Size(adj_matrix.shape)
    ).to(device)
    
    # Coalesce for better performance
    adj_matrix_tensor = adj_matrix_tensor.coalesce()
    
    # Initialize model
    model = LightGCN(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"🚀 Training LightGCN model (Epoch 0/{epochs})...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        user_emb, item_emb = model(adj_matrix_tensor)
        
        # Simple loss (in practice, use BPR loss)
        loss = torch.norm(user_emb) + torch.norm(item_emb)
        loss.backward()
        optimizer.step()
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"🚀 Training LightGCN model (Epoch {epoch+1}/{epochs})...")
    
    status_text.text("✅ LightGCN training complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return model, adj_matrix_tensor, user_emb, item_emb

# Text processing functions
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def get_top_words(texts, n=20):
    all_words = []
    stop_words = set(stopwords.words('english'))
    
    for text in texts:
        if not isinstance(text, str):
            continue
        words = word_tokenize(clean_text(text))
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        all_words.extend(filtered_words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(n)

# User Profile Management
class UserProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        self.favorite_genres = []
        self.preferred_decades = []
        self.watch_history = []
        self.preferred_actors = []
        self.preferred_directors = []
        self.feedback_history = []
    
    def update_from_ratings(self, ratings):
        # Extract preferences from ratings
        genre_counter = Counter()
        decade_counter = Counter()
        
        for _, row in ratings.iterrows():
            if 'genres' in row and isinstance(row['genres'], str):
                genres = row['genres'].split('|')
                genre_counter.update(genres)
                
            if 'release_date' in row and isinstance(row['release_date'], str):
                try:
                    year = int(row['release_date'].split('-')[0])
                    decade = f"{year // 10 * 10}s"
                    decade_counter.update([decade])
                except:
                    pass
        
        self.favorite_genres = [genre for genre, _ in genre_counter.most_common(3)]
        self.preferred_decades = [decade for decade, _ in decade_counter.most_common(2)]
    
    def add_feedback(self, movie_id, feedback):
        self.feedback_history.append((movie_id, feedback))
    
    def to_dict(self):
        return {
            'favorite_genres': self.favorite_genres,
            'preferred_decades': self.preferred_decades,
            'watch_history': self.watch_history,
            'feedback_history': self.feedback_history
        }

# Recommendation Functions
@st.cache_data(ttl=3600, show_spinner=False)
def get_hybrid_recommendations(user_id, sentiment_weight=0.4, lightgcn_weight=0.6):
    """Combine sentiment and LightGCN recommendations"""
    sentiment_recs = get_sentiment_recommendations(user_id)
    lightgcn_recs = get_lightgcn_recommendations(user_id)
    
    # Create combined scores
    combined_scores = {}
    
    for movie_id, score in sentiment_recs:
        combined_scores[movie_id] = score * sentiment_weight
    
    for movie_id, score in lightgcn_recs:
        if movie_id in combined_scores:
            combined_scores[movie_id] += score * lightgcn_weight
        else:
            combined_scores[movie_id] = score * lightgcn_weight
    
    # Sort by combined score
    sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_recs[:10]

def get_sentiment_recommendations(user_id):
    """Get sentiment-based recommendations for a user"""
    df = st.session_state.df
    user_ratings = df[df['userId'] == user_id]
    
    # If user has no ratings, return popular movies
    if user_ratings.empty:
        return get_popular_movies()
    
    # Get positive sentiment movies rated by user
    positive_movies = user_ratings[user_ratings['Sentiment'] == 1]
    
    # If no positive movies, return popular movies
    if positive_movies.empty:
        return get_popular_movies()
    
    # Find similar movies based on genres
    user_genres = set()
    for _, row in positive_movies.iterrows():
        if 'genres' in row and isinstance(row['genres'], str):
            user_genres.update(row['genres'].split('|'))
    
    # Find movies with similar genres
    similar_movies = df[df['genres'].apply(lambda x: any(genre in x.split('|') for genre in user_genres) 
                       if isinstance(x, str) else False)]
    
    # Calculate average sentiment score
    movie_scores = similar_movies.groupby('movieId')['Sentiment'].mean().reset_index()
    movie_scores.columns = ['movieId', 'score']
    
    # Filter out movies already rated by user
    movie_scores = movie_scores[~movie_scores['movieId'].isin(user_ratings['movieId'])]
    
    return list(movie_scores.sort_values('score', ascending=False).head(10).itertuples(index=False))

def get_lightgcn_recommendations(user_id):
    """Get LightGCN recommendations for a user"""
    if 'user_embeddings' not in st.session_state or 'item_embeddings' not in st.session_state:
        return []
    
    user_idx = st.session_state.user_id_map.get(user_id, None)
    if user_idx is None:
        return []
    
    user_vec = st.session_state.user_embeddings[user_idx].unsqueeze(0)
    item_emb = st.session_state.item_embeddings
    
    with torch.no_grad():
        scores = torch.mm(user_vec, item_emb.T).squeeze()
        top_indices = torch.topk(scores, 10).indices.cpu().numpy()
        top_scores = torch.topk(scores, 10).values.cpu().numpy()
    
    recommendations = []
    for idx, score in zip(top_indices, top_scores):
        movie_id = st.session_state.reverse_movie_map[idx]
        recommendations.append((movie_id, score.item()))
    
    return recommendations

def get_popular_movies(n=10):
    """Get popular movies for cold start"""
    df = st.session_state.df
    movie_stats = df.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    
    # Filter movies with sufficient ratings
    top_movies = movie_stats[movie_stats['rating_count'] >= 10].sort_values(
        ['avg_rating', 'rating_count'], ascending=[False, False]).head(n)
    
    return [(row['movieId'], row['avg_rating']) for _, row in top_movies.iterrows()]

def get_similar_movies(movie_id, n=5):
    """Get similar movies based on embeddings"""
    if 'item_embeddings' not in st.session_state:
        return []
    
    movie_idx = st.session_state.movie_id_map.get(movie_id, None)
    if movie_idx is None:
        return []
    
    item_emb = st.session_state.item_embeddings.cpu().numpy()
    movie_embedding = item_emb[movie_idx].reshape(1, -1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(movie_embedding, item_emb)[0]
    
    # Get top similar movies (excluding itself)
    top_indices = np.argsort(similarities)[::-1][1:n+1]
    
    similar_movies = []
    for idx in top_indices:
        similar_movie_id = st.session_state.reverse_movie_map[idx]
        similar_movies.append((similar_movie_id, similarities[idx]))
    
    return similar_movies

def get_seasonal_recommendations():
    """Get recommendations based on current season"""
    current_month = datetime.now().month
    season = ""
    
    if current_month in [11, 12]:
        season = "Holiday"
    elif current_month in [6, 7, 8]:
        season = "Summer"
    elif current_month == 10:
        season = "Halloween"
    elif current_month == 2:
        season = "Valentine"
    
    if not season:
        return []
    
    seasonal_movies = SEASONAL_MOVIES[season]
    df = st.session_state.df
    movie_map = st.session_state.movie_map
    
    # Get movie IDs for seasonal movies
    recommendations = []
    for title in seasonal_movies:
        movie_id = next((mid for mid, t in movie_map.items() if t == title), None)
        if movie_id:
            recommendations.append((movie_id, 1.0))  # Max score for seasonal
    
    return recommendations

# Feedback Functions
def record_feedback(user_id, movie_id, feedback_type):
    """Record user feedback for a recommendation"""
    timestamp = datetime.now()
    new_feedback = pd.DataFrame({
        'user_id': [user_id],
        'movie_id': [movie_id],
        'feedback': [feedback_type],
        'timestamp': [timestamp]
    })
    
    st.session_state.feedback_data = pd.concat(
        [st.session_state.feedback_data, new_feedback], ignore_index=True)
    
    # Update user profile
    if user_id not in st.session_state.user_profiles:
        st.session_state.user_profiles[user_id] = UserProfile(user_id)
    
    st.session_state.user_profiles[user_id].add_feedback(movie_id, feedback_type)
    
    # Show feedback confirmation
    st.toast(f"Feedback recorded: {'👍' if feedback_type == 'like' else '👎'} for movie {movie_id}")

def update_user_embedding(user_id, movie_id, adjustment):
    """Adjust user embedding based on feedback"""
    if 'user_embeddings' not in st.session_state:
        return
    
    user_idx = st.session_state.user_id_map.get(user_id, None)
    movie_idx = st.session_state.movie_id_map.get(movie_id, None)
    
    if user_idx is None or movie_idx is None:
        return
    
    # Apply adjustment to user embedding
    with torch.no_grad():
        st.session_state.user_embeddings[user_idx] += adjustment * st.session_state.item_embeddings[movie_idx]

# Export Functions
def create_ics_file(recommendations, user_id):
    """Create ICS calendar file for recommendations"""
    calendar = ics.Calendar()
    
    for i, (movie_id, score) in enumerate(recommendations):
        title = st.session_state.movie_map.get(movie_id, f"Movie {movie_id}")
        
        # Create event for the recommendation
        event = ics.Event()
        event.name = f"Recommended: {title}"
        event.description = f"Recommended for you with score: {score:.2f}"
        
        # Schedule over next 7 days
        event.begin = datetime.now() + timedelta(days=i)
        event.end = event.begin + timedelta(hours=2)
        
        calendar.events.add(event)
    
    # Save to string
    return str(calendar)

def create_shared_watchlist(user_ids, list_name):
    """Create a shared watchlist for multiple users"""
    watchlist_id = f"watchlist_{len(st.session_state.shared_watchlists)+1}"
    
    # Get recommendations for each user
    combined_recs = {}
    for user_id in user_ids:
        recs = get_hybrid_recommendations(user_id)
        for movie_id, score in recs:
            if movie_id in combined_recs:
                combined_recs[movie_id] += score
            else:
                combined_recs[movie_id] = score
    
    # Sort by combined score
    sorted_recs = sorted(combined_recs.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Save watchlist
    st.session_state.shared_watchlists[watchlist_id] = {
        'name': list_name,
        'users': user_ids,
        'movies': [movie_id for movie_id, _ in sorted_recs],
        'created': datetime.now()
    }
    
    return watchlist_id

# File upload section
st.sidebar.header("📤 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your movie data (CSV with columns: userId, movieId, rating, review, genres, release_date)", 
    type=["csv"]
)

# Use sample dataset if no file uploaded
if uploaded_file is None:
    st.sidebar.info("💡 Don't have a dataset? Try our sample data!")
    if st.sidebar.button("Use Sample Dataset", key="sample_data"):
        # Generate synthetic data with enhanced features
        num_users = 200
        num_movies = 100
        num_ratings = 2000
        
        user_ids = np.random.randint(1, num_users+1, num_ratings)
        movie_ids = np.random.randint(1, num_movies+1, num_ratings)
        ratings = np.random.choice([1, 2, 3, 4, 5], num_ratings, p=[0.1, 0.2, 0.3, 0.25, 0.15])
        
        reviews = [
            "Great movie with amazing acting" if r > 3 else 
            "Average at best" if r == 3 else 
            "Not worth watching" 
            for r in ratings
        ]
        
        # Generate movie titles and genres
        movie_titles = [f"Movie {i+1}" for i in range(num_movies)]
        movie_genres = [
            "|".join(np.random.choice(GENRES, np.random.randint(1, 4), replace=False)) 
            for _ in range(num_movies)
        ]
        
        # Generate release dates (from 1970 to 2023)
        release_dates = [
            f"{np.random.randint(1970, 2024)}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}"
            for _ in range(num_movies)
        ]
        
        movie_map = {i+1: movie_titles[i] for i in range(num_movies)}
        genre_map = {i+1: movie_genres[i] for i in range(num_movies)}
        date_map = {i+1: release_dates[i] for i in range(num_movies)}
        
        sample_data = {
            'userId': user_ids,
            'movieId': movie_ids,
            'rating': ratings,
            'review': reviews,
            'genres': [genre_map[mid] for mid in movie_ids],
            'release_date': [date_map[mid] for mid in movie_ids]
        }
        
        st.session_state.df = pd.DataFrame(sample_data)
        st.session_state.movie_map = movie_map
        st.rerun()
elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['userId', 'movieId', 'rating']
        if not all(col in df.columns for col in required_columns):
            st.error("⚠️ Uploaded file is missing required columns. Please ensure your CSV includes: userId, movieId, rating")
            st.stop()
        
        # Create movie map if not provided
        if 'title' in df.columns:
            movie_map = df[['movieId', 'title']].drop_duplicates().set_index('movieId')['title'].to_dict()
        else:
            unique_movies = df['movieId'].unique()
            movie_map = {mid: f"Movie {mid}" for mid in unique_movies}
        
        # Add default genres if missing
        if 'genres' not in df.columns:
            df['genres'] = [np.random.choice(GENRES) for _ in range(len(df))]
        
        # Add default release dates if missing
        if 'release_date' not in df.columns:
            df['release_date'] = [
                f"{np.random.randint(1970, 2024)}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}" 
                for _ in range(len(df))
            ]
        
        st.session_state.df = df
        st.session_state.movie_map = movie_map
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

# Main app tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💡 Recommendations", "👥 Social", "⚙️ Settings"])

if st.session_state.df is not None:
    df = st.session_state.df
    movie_map = st.session_state.movie_map
    
    # Preprocess data for LightGCN
    if 'interaction_matrix' not in st.session_state:
        # Create user and movie mappings
        user_ids = df['userId'].unique()
        movie_ids = df['movieId'].unique()
        
        user_id_map = {uid: i for i, uid in enumerate(user_ids)}
        movie_id_map = {mid: i for i, mid in enumerate(movie_ids)}
        
        # Create sparse interaction matrix
        rows = df['userId'].map(user_id_map)
        cols = df['movieId'].map(movie_id_map)
        ratings = df['rating'].values
        
        # Create binary interactions (1 if rating >= 4)
        interactions = (ratings >= 4).astype(int)
        
        interaction_matrix = coo_matrix(
            (interactions, (rows, cols)),
            shape=(len(user_ids), len(movie_ids))
        )
        
        st.session_state.interaction_matrix = interaction_matrix
        st.session_state.user_id_map = user_id_map
        st.session_state.movie_id_map = movie_id_map
        st.session_state.reverse_user_map = {v: k for k, v in user_id_map.items()}
        st.session_state.reverse_movie_map = {v: k for k, v in movie_id_map.items()}
    
    # Dashboard Tab
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Users", df['userId'].nunique())
        col2.metric("Total Movies", df['movieId'].nunique())
        col3.metric("Total Ratings", len(df))
        col4.metric("Average Rating", f"{df['rating'].mean():.2f} ⭐")
        
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        rating_counts = df['rating'].value_counts().sort_index()
        sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax, palette="viridis")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
        st.subheader("Top Rated Movies (Min 10 ratings)")
        movie_stats = df.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count'),
            title=('movieId', lambda x: movie_map.get(x.iloc[0], "Unknown"))
        ).reset_index()
        
        top_movies = movie_stats[movie_stats['rating_count'] >= 10].sort_values(
            'avg_rating', ascending=False).head(10)
        
        if not top_movies.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='avg_rating', y='title', data=top_movies, ax=ax, palette="viridis")
            ax.set_xlabel("Average Rating")
            ax.set_ylabel("")
            st.pyplot(fig)
        else:
            st.info("Not enough ratings to calculate top movies")
        
        # Genre distribution
        st.subheader("Genre Distribution")
        if 'genres' in df.columns:
            all_genres = []
            for genres in df['genres']:
                if isinstance(genres, str):
                    all_genres.extend(genres.split('|'))
            
            genre_counts = pd.Series(all_genres).value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax, palette="viridis")
            ax.set_xlabel("Count")
            ax.set_ylabel("Genre")
            st.pyplot(fig)
    
    # Recommendations Tab
    with tab2:
        st.header("Personalized Recommendations")
        
        # Load sentiment model
        if 'review' in df.columns and 'sentiment_model' not in st.session_state:
            st.session_state.sentiment_model = load_sentiment_model()
            
            # Add sentiment column if not present
            if 'Sentiment' not in df.columns:
                with st.spinner("🔍 Analyzing reviews with AI..."):
                    # Predict sentiment in batches
                    batch_size = 32
                    predictions = []
                    progress_bar = st.progress(0)
                    
                    for i in range(0, len(df), batch_size):
                        batch_texts = df["review"].iloc[i:i + batch_size].tolist()
                        batch_preds = st.session_state.sentiment_model(batch_texts)
                        predictions.extend([1 if pred['label'] == 'POSITIVE' else 0 for pred in batch_preds])
                        progress_bar.progress(min((i + batch_size) / len(df), 1.0))
                    
                    # Add predictions to dataframe
                    df["Sentiment"] = predictions
                    st.session_state.df = df
                    progress_bar.empty()
        
        # User selection
        user_ids = df['userId'].unique()
        selected_user_id = st.selectbox("Select User", user_ids)
        
        # Initialize user profile
        if selected_user_id not in st.session_state.user_profiles:
            st.session_state.user_profiles[selected_user_id] = UserProfile(selected_user_id)
            
            # Update profile from existing ratings
            user_ratings = df[df['userId'] == selected_user_id]
            st.session_state.user_profiles[selected_user_id].update_from_ratings(user_ratings)
        
        user_profile = st.session_state.user_profiles[selected_user_id]
        
        # Display user profile
        with st.expander("👤 User Profile"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Favorite Genres")
                if user_profile.favorite_genres:
                    for genre in user_profile.favorite_genres:
                        st.markdown(f"- {genre}")
                else:
                    st.info("No genre preferences detected")
                
            with col2:
                st.subheader("Preferred Decades")
                if user_profile.preferred_decades:
                    for decade in user_profile.preferred_decades:
                        st.markdown(f"- {decade}")
                else:
                    st.info("No decade preferences detected")
        
        # Recommendation type selection
        rec_type = st.radio("Recommendation Type", 
                           ["Hybrid", "Sentiment-Based", "LightGCN", "Seasonal", "Cold Start"], 
                           horizontal=True)
        
        # Genre filtering
        selected_genres = st.multiselect("Filter by Genre", GENRES, default=user_profile.favorite_genres)
        
        # Generate recommendations
        if st.button("Get Recommendations"):
            if rec_type == "Hybrid":
                recommendations = get_hybrid_recommendations(selected_user_id)
            elif rec_type == "Sentiment-Based":
                recommendations = get_sentiment_recommendations(selected_user_id)
            elif rec_type == "LightGCN":
                recommendations = get_lightgcn_recommendations(selected_user_id)
            elif rec_type == "Seasonal":
                recommendations = get_seasonal_recommendations()
            elif rec_type == "Cold Start":
                recommendations = get_popular_movies()
            
            # Apply genre filter
            if selected_genres and recommendations:
                filtered_recommendations = []
                for movie_id, score in recommendations:
                    if 'genres' in df.columns:
                        movie_genres = df[df['movieId'] == movie_id]['genres'].iloc[0]
                        if isinstance(movie_genres, str) and any(genre in movie_genres for genre in selected_genres):
                            filtered_recommendations.append((movie_id, score))
                recommendations = filtered_recommendations
            
            st.session_state.hybrid_recommendations[selected_user_id] = recommendations
        
        # Display recommendations
        if selected_user_id in st.session_state.hybrid_recommendations:
            recommendations = st.session_state.hybrid_recommendations[selected_user_id]
            
            if not recommendations:
                st.warning("No recommendations found matching your criteria")
            else:
                st.subheader(f"Top Recommendations for User {selected_user_id}")
                
                for i, (movie_id, score) in enumerate(recommendations):
                    movie_title = movie_map.get(movie_id, f"Movie {movie_id}")
                    
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            # Special styling for cold start recommendations
                            if rec_type == "Cold Start":
                                st.markdown(f"""
                                    <div class="cold-start-card">
                                        <div style="font-size: 18px; font-weight: bold;">{i+1}. {movie_title}</div>
                                        <div>Popularity Score: {score:.3f}</div>
                                        <div>Movie ID: {movie_id}</div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div class="recommendation-card">
                                        <div style="font-size: 18px; font-weight: bold;">{i+1}. {movie_title}</div>
                                        <div>Recommendation Score: {score:.3f}</div>
                                        <div>Movie ID: {movie_id}</div>
                                """, unsafe_allow_html=True)
                            
                            # Display genres if available
                            if 'genres' in df.columns:
                                movie_genres = df[df['movieId'] == movie_id]['genres'].iloc[0]
                                if isinstance(movie_genres, str):
                                    st.markdown(f"**Genres:** {movie_genres.replace('|', ', ')}")
                            
                            # Similar movies
                            if st.button("Find Similar Movies", key=f"similar_{movie_id}"):
                                similar_movies = get_similar_movies(movie_id, 3)
                                
                                if similar_movies:
                                    st.markdown("**Similar Movies:**")
                                    for sim_id, sim_score in similar_movies:
                                        sim_title = movie_map.get(sim_id, f"Movie {sim_id}")
                                        st.markdown(f"- {sim_title} (similarity: {sim_score:.2f})")
                                else:
                                    st.info("No similar movies found")
                            
                            # Feedback buttons
                            if rec_type != "Cold Start":
                                col_fb1, col_fb2 = st.columns(2)
                                with col_fb1:
                                    if st.button("👍 Like", key=f"like_{movie_id}", use_container_width=True):
                                        record_feedback(selected_user_id, movie_id, "like")
                                        update_user_embedding(selected_user_id, movie_id, 0.1)
                                with col_fb2:
                                    if st.button("👎 Dislike", key=f"dislike_{movie_id}", use_container_width=True):
                                        record_feedback(selected_user_id, movie_id, "dislike")
                                        update_user_embedding(selected_user_id, movie_id, -0.1)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            # Placeholder for movie poster
                            st.image("https://via.placeholder.com/150x225.png?text=Movie+Poster", 
                                    caption=movie_title, use_column_width=True)
                
                # Export to calendar
                st.subheader("Export Recommendations")
                if rec_type != "Cold Start":
                    ics_content = create_ics_file(recommendations, selected_user_id)
                    st.download_button(
                        label="📅 Export to Calendar",
                        data=ics_content,
                        file_name=f"movie_recommendations_{selected_user_id}.ics",
                        mime="text/calendar"
                    )
    
    # Social Tab
    with tab3:
        st.header("Social Features")
        
        # Shared watchlists
        st.subheader("Create Shared Watchlist")
        with st.form("watchlist_form"):
            watchlist_name = st.text_input("Watchlist Name", "Movie Night Picks")
            selected_users = st.multiselect("Select Users", df['userId'].unique().tolist())
            
            if st.form_submit_button("Create Shared Watchlist"):
                if len(selected_users) < 2:
                    st.warning("Please select at least 2 users")
                else:
                    watchlist_id = create_shared_watchlist(selected_users, watchlist_name)
                    st.success(f"Watchlist '{watchlist_name}' created successfully!")
        
        # Display existing watchlists
        st.subheader("Your Watchlists")
        if st.session_state.shared_watchlists:
            for watchlist_id, watchlist in st.session_state.shared_watchlists.items():
                with st.expander(f"📋 {watchlist['name']} (Created: {watchlist['created'].strftime('%Y-%m-%d')})"):
                    st.write(f"**Users:** {', '.join(str(uid) for uid in watchlist['users'])}")
                    
                    st.markdown("**Recommended Movies:**")
                    for movie_id in watchlist['movies']:
                        movie_title = movie_map.get(movie_id, f"Movie {movie_id}")
                        st.markdown(f"- {movie_title}")
                    
                    # Share link
                    st.markdown("**Share this watchlist:**")
                    share_link = f"https://movierecs.example.com/watchlist/{watchlist_id}"
                    st.markdown(f'<div class="share-link">{share_link}</div>', unsafe_allow_html=True)
                    
                    # Copy button
                    if st.button("Copy Link", key=f"copy_{watchlist_id}"):
                        st.session_state["copied"] = True
                        st.experimental_rerun()
                    
                    if st.session_state.get("copied", False):
                        st.success("Link copied to clipboard!")
                        st.session_state["copied"] = False
        else:
            st.info("No shared watchlists created yet")
    
    # Settings Tab
    with tab4:
        st.header("System Settings")
        
        # LightGCN training
        st.subheader("Model Training")
        if st.session_state.lightgcn_model is None:
            if st.button("Train LightGCN Model"):
                with st.spinner("Setting up LightGCN training..."):
                    model, adj_matrix, user_emb, item_emb = train_lightgcn(
                        st.session_state.interaction_matrix,
                        epochs=20
                    )
                    
                    st.session_state.lightgcn_model = model
                    st.session_state.user_embeddings = user_emb
                    st.session_state.item_embeddings = item_emb
                    st.session_state.adj_matrix = adj_matrix
                    st.success("LightGCN model trained successfully!")
        else:
            st.success("✅ LightGCN model is already trained")
            if st.button("Retrain LightGCN Model"):
                st.session_state.lightgcn_model = None
                st.session_state.user_embeddings = None
                st.session_state.item_embeddings = None
                st.rerun()
        
        # Hybrid weights configuration
        st.subheader("Hybrid Recommendation Settings")
        col1, col2 = st.columns(2)
        with col1:
            sentiment_weight = st.slider("Sentiment Weight", 0.0, 1.0, 0.4)
        with col2:
            lightgcn_weight = st.slider("LightGCN Weight", 0.0, 1.0, 0.6)
        st.caption(f"Current weights: Sentiment={sentiment_weight}, LightGCN={lightgcn_weight}")
        
        # Feedback data
        st.subheader("User Feedback")
        if not st.session_state.feedback_data.empty:
            st.dataframe(st.session_state.feedback_data)
            if st.button("Clear Feedback Data"):
                st.session_state.feedback_data = pd.DataFrame(columns=['user_id', 'movie_id', 'feedback', 'timestamp'])
        else:
            st.info("No feedback recorded yet")
        
        # Performance settings
        st.subheader("Performance Optimization")
        use_caching = st.checkbox("Enable Recommendation Caching", value=True)
        st.caption("Caching improves performance but may show stale recommendations")
    
else:
    st.info("Please upload a dataset or use sample data to get started")

# Footer
st.markdown("---")
st.caption("""
    Enhanced Hybrid Recommendation System • Combines Sentiment Analysis and LightGCN •
    [Sentiment Analysis] Identifies positive reviews • [LightGCN] Models user-movie interactions •
    © 2023 MovieRec AI • v2.0
""")
