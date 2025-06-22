# Deep-Dive-into-Neural-Recommendation-Systems

This project implements a sophisticated hybrid movie recommendation system that combines sentiment analysis of user reviews with LightGCN graph neural networks. The system provides personalized movie recommendations through an interactive Streamlit interface.

Features
🎭 Dual Recommendation Approaches
Sentiment-Based Recommendations:

Analyzes review text using DistilBERT NLP model

Scores movies based on positive sentiment frequency and strength

LightGCN Recommendations:

Implements graph neural network for collaborative filtering

Learns user-movie interactions through message passing



📊 Comprehensive Analytics
User and movie statistics dashboard

Rating distribution visualizations

Sentiment analysis with word frequency charts

Top movie recommendations with scoring metrics



💻 User-Friendly Interface
Tab-based navigation system

Interactive recommendation cards

Requirements
Python 3.8+

Streamlit

PyTorch

Transformers

Pandas

NumPy

SciPy

Matplotlib

Seaborn

NLTK

Technologies Used

🤖 Machine Learning
DistilBERT: For sentiment analysis of reviews

LightGCN: Graph neural network for collaborative filtering

PyTorch: Deep learning framework
Progress indicators for long operations

Sample dataset for quick testing

Responsive design for all screen sizes


📊 Data Processing
Pandas for data manipulation

SciPy for sparse matrix operations

NLTK for text processing

🎨 Frontend
Streamlit for web interface

Matplotlib/Seaborn for visualizations

Custom CSS for styling



Methodology
Sentiment Analysis Pipeline
Text cleaning and preprocessing

Batch processing with DistilBERT

Sentiment classification (positive/negative)

Movie scoring based on:

Frequency of positive reviews (70% weight)

Average sentiment strength (30% weight)

LightGCN Implementation
Construct user-movie interaction graph

Create sparse adjacency matrix

Initialize user and movie embeddings

Perform graph convolution:

python
for _ in range(n_layers):
    embeddings = torch.sparse.mm(adj_matrix, embeddings)
Combine embeddings from all layers

Generate recommendations through dot product scoring
