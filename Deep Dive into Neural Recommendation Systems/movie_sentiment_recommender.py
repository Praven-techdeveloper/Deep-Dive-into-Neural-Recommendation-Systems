import os
os.environ['USE_TORCH'] = '1'  # Force transformers to use PyTorch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'  # Disable warnings

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

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set up the page
st.set_page_config(
    page_title="Hybrid Movie Recommender", 
    layout="wide",
    page_icon="🎬"
)

st.title("🎬 Hybrid Movie Recommendation System")
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

# Fixed sentiment model loader (using PyTorch)
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
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def get_top_words(texts, n=20):
    all_words = []
    stop_words = set(stopwords.words('english'))
    
    for text in texts:
        words = word_tokenize(clean_text(text))
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        all_words.extend(filtered_words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(n)

# File upload section
st.sidebar.header("📤 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your movie data (CSV with columns: userId, movieId, rating, review)", 
    type=["csv"]
)

# Use sample dataset if no file uploaded
if uploaded_file is None:
    st.sidebar.info("💡 Don't have a dataset? Try our sample data!")
    if st.sidebar.button("Use Sample Dataset", key="sample_data"):
        # Generate synthetic data
        num_users = 200
        num_movies = 100
        num_ratings = 2000
        
        user_ids = np.random.randint(0, num_users, num_ratings)
        movie_ids = np.random.randint(0, num_movies, num_ratings)
        ratings = np.random.choice([1, 2, 3, 4, 5], num_ratings, p=[0.1, 0.2, 0.3, 0.25, 0.15])
        
        reviews = [
            "Great movie with amazing acting" if r > 3 else 
            "Average at best" if r == 3 else 
            "Not worth watching" 
            for r in ratings
        ]
        
        movies = [f"Movie {i+1}" for i in range(num_movies)]
        movie_map = {i: movies[i] for i in range(num_movies)}
        
        sample_data = {
            'userId': user_ids,
            'movieId': movie_ids,
            'rating': ratings,
            'review': reviews
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
        
        st.session_state.df = df
        st.session_state.movie_map = movie_map
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

# Main app tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💡 Sentiment Analysis", "🧠 LightGCN Recommendations"])

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
            rating_count=('rating', 'count')
        ).reset_index()
        
        top_movies = movie_stats[movie_stats['rating_count'] >= 10].sort_values(
            'avg_rating', ascending=False).head(10)
        
        if not top_movies.empty:
            top_movies['title'] = top_movies['movieId'].map(movie_map)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='avg_rating', y='title', data=top_movies, ax=ax, palette="viridis")
            ax.set_xlabel("Average Rating")
            ax.set_ylabel("")
            st.pyplot(fig)
        else:
            st.info("Not enough ratings to calculate top movies")
    
    # Sentiment Analysis Tab
    with tab2:
        st.header("Sentiment-Based Recommendations")
        
        if 'review' not in df.columns:
            st.warning("No review column found. Sentiment analysis requires review text.")
        else:
            if 'sentiment_model' not in st.session_state:
                st.session_state.sentiment_model = load_sentiment_model()
            
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
                    st.session_state.sentiment_recommendations = True
                    progress_bar.empty()
            
            # Sentiment distribution
            st.subheader("Sentiment Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(6, 6))
                sentiment_counts = df['Sentiment'].value_counts()
                labels = ['Positive' if i == 1 else 'Negative' for i in sentiment_counts.index]
                colors = ['#51cf66', '#ff6b6b']
                
                sentiment_counts.plot.pie(
                    autopct='%1.1f%%', 
                    labels=labels,
                    colors=colors,
                    shadow=True,
                    startangle=90,
                    ax=ax
                )
                ax.set_ylabel('')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Top Words in Positive Reviews")
                positive_texts = df[df['Sentiment'] == 1]['review'].tolist()
                
                if positive_texts:
                    top_words = get_top_words(positive_texts, 10)
                    words, counts = zip(*top_words)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=list(counts), y=list(words), palette="viridis", ax=ax)
                    ax.set_title("Most Frequent Positive Words")
                    ax.set_xlabel("Count")
                    ax.set_ylabel("")
                    st.pyplot(fig)
                else:
                    st.info("No positive reviews to analyze")
            
            # Sentiment-based recommendations
            st.subheader("Top Recommended Movies by Sentiment")
            positive_reviews = df[df['Sentiment'] == 1]
            
            if not positive_reviews.empty:
                movie_stats = positive_reviews.groupby('movieId').agg(
                    positive_reviews=('Sentiment', 'count'),
                    avg_sentiment=('Sentiment', 'mean')
                ).reset_index()
                
                # Normalize and score
                max_reviews = movie_stats['positive_reviews'].max()
                movie_stats['score'] = 0.7 * (movie_stats['positive_reviews'] / max_reviews) + 0.3 * movie_stats['avg_sentiment']
                
                # Get top recommendations
                top_movies = movie_stats.sort_values('score', ascending=False).head(5)
                top_movies['title'] = top_movies['movieId'].map(movie_map)
                
                for _, row in top_movies.iterrows():
                    with st.container():
                        st.markdown(f"""
                            <div class="recommendation-card">
                                <div style="font-size: 18px; font-weight: bold;">{row['title']}</div>
                                <div>Sentiment Score: {row['score']:.3f}</div>
                                <div>Positive Reviews: {row['positive_reviews']}</div>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No positive reviews found for recommendations")
    
    # LightGCN Recommendations Tab
    with tab3:
        st.header("LightGCN Recommendations")
        st.markdown("""
            <div class="model-card">
                <b>LightGCN (Graph Convolutional Network)</b> learns user and movie embeddings by propagating 
                interactions over the user-movie graph. It captures collaborative filtering signals without 
                heavy feature engineering.
            </div>
        """, unsafe_allow_html=True)
        
        # Train LightGCN if not already trained
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
        
        # Show recommendations if model is trained
        if st.session_state.lightgcn_model is not None:
            st.success("✅ LightGCN model is trained and ready for recommendations!")
            
            # Select user
            user_ids = list(st.session_state.reverse_user_map.keys())
            selected_user_idx = st.selectbox("Select User", options=user_ids, format_func=lambda x: f"User {st.session_state.reverse_user_map[x]}")
            
            # Get recommendations
            if st.button("Generate Recommendations"):
                model = st.session_state.lightgcn_model
                item_emb = st.session_state.item_embeddings
                
                with torch.no_grad():
                    user_vec = st.session_state.user_embeddings[selected_user_idx].unsqueeze(0)
                    scores = torch.mm(user_vec, item_emb.T).squeeze()
                    
                    # Get top scores
                    top_indices = torch.topk(scores, 10).indices.cpu().numpy()
                    top_scores = torch.topk(scores, 10).values.cpu().numpy()
                
                st.subheader(f"Top Recommendations for User {st.session_state.reverse_user_map[selected_user_idx]}")
                
                # Display recommendations
                for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                    movie_id = st.session_state.reverse_movie_map[idx]
                    movie_title = movie_map.get(movie_id, f"Movie {movie_id}")
                    
                    with st.container():
                        st.markdown(f"""
                            <div class="recommendation-card">
                                <div style="font-size: 18px; font-weight: bold;">{i+1}. {movie_title}</div>
                                <div>Match Score: {score:.3f}</div>
                                <div>Movie ID: {movie_id}</div>
                            </div>
                        """, unsafe_allow_html=True)
        
        else:
            st.info("Click the 'Train LightGCN Model' button to start training")
    
else:
    st.info("Please upload a dataset or use sample data to get started")

# Footer
st.markdown("---")
st.caption("""
    Hybrid Recommendation System • Combines Sentiment Analysis and LightGCN •
    [Sentiment Analysis] Identifies positive reviews • [LightGCN] Models user-movie interactions
""")
