import streamlit as st
import pandas as pd
import pickle
from recommendation_engine import MovieRecommender
import plotly.express as px
import plotly.graph_objects as go

# Page configuration 
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_recommender():
    recommender = MovieRecommender()
    try:
        # Get modification time of the model to handle caching
        import os
        model_path = 'movie_recommender.pkl'
        mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0
        
        recommender.load_model(model_path)
        return recommender
    except:
        st.error("⚠️ Model not found! Please run 'recommendation_engine.py' first.")
        return None

# Title
st.title("🎬 IMDb Movie Recommendation System")
st.markdown("### Find movies similar to your favorite storylines!")

# Sidebar
with st.sidebar:
    st.header("📊 About")
    st.info("""
    This AI-powered system uses:
    - **NLP** for text processing
    - **TF-IDF** for feature extraction
    - **Cosine Similarity** for recommendations
    
    Built with Python, Scikit-learn & Streamlit
    """)
    
    st.header("📈 Statistics")
    recommender = load_recommender()
    if recommender:
        st.metric("Total Movies", len(recommender.df))
        st.metric("Features", recommender.tfidf_matrix.shape[1])

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Find Similar", "📝 Custom Storyline", "📊 Data Insights"])

# Tab 1: Find Similar Movies
with tab1:
    st.header("Find Movies Similar to...")
    
    recommender = load_recommender()
    if recommender:
        # Movie selection
        movie_list = recommender.df['Movie_Name'].tolist()
        selected_movie = st.selectbox(
            "Select a movie:",
            options=movie_list,
            index=0
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            top_n = st.slider("Number of recommendations:", 1, 10, 5)
        with col2:
            if st.button("🎯 Get Recommendations", type="primary"):
                with st.spinner("Finding similar movies..."):
                    recommendations = recommender.find_similar_movies(selected_movie, top_n)
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} similar movies!")
                        
                        # Display selected movie info
                        st.subheader("📽️ Selected Movie")
                        movie_data = recommender.df[recommender.df['Movie_Name'] == selected_movie].iloc[0]
                        st.info(f"**{selected_movie}**\n\n{movie_data['Storyline']}")
                        
                        # Display recommendations
                        st.subheader("✨ Recommended Movies")
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"#{i} - {rec['Movie_Name']} (Match: {rec['Similarity_Score']}%)"):
                                st.write(f"**Storyline:** {rec['Storyline']}")
                                
                                # Progress bar for similarity
                                st.progress(rec['Similarity_Score'] / 100)
                        
                        # Visualization
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[rec['Movie_Name'] for rec in recommendations],
                                y=[rec['Similarity_Score'] for rec in recommendations],
                                marker_color='lightblue'
                            )
                        ])
                        fig.update_layout(
                            title="Similarity Scores",
                            xaxis_title="Movie",
                            yaxis_title="Similarity (%)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Movie not found!")

# Tab 2: Custom Storyline Input
with tab2:
    st.header("Enter Your Own Storyline")
    
    recommender = load_recommender()
    if recommender:
        user_storyline = st.text_area(
            "Describe a movie plot:",
            placeholder="Example: A young wizard begins his journey at a magical school where he makes friends and enemies, facing dark forces along the way.",
            height=150
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            top_n2 = st.slider("Top N movies:", 1, 10, 5, key="slider2")
        
        if st.button("🔮 Find Similar Movies", type="primary"):
            if user_storyline.strip():
                with st.spinner("Analyzing storyline..."):
                    recommendations = recommender.find_similar_by_storyline(user_storyline, top_n2)
                    
                    st.success(f"Found {len(recommendations)} matching movies!")
                    
                    st.subheader("✨ Top Recommendations")
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"#{i} - {rec['Movie_Name']} (Match: {rec['Similarity_Score']}%)"):
                            st.write(f"**Storyline:** {rec['Storyline']}")
                            st.progress(rec['Similarity_Score'] / 100)
            else:
                st.warning("⚠️ Please enter a storyline!")

# Tab 3: Data Insights
with tab3:
    st.header("Dataset Insights")
    
    recommender = load_recommender()
    if recommender:
        df = recommender.df
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Movies", len(df))
        with col2:
            avg_length = df['Storyline'].str.len().mean()
            st.metric("Avg Storyline Length", f"{int(avg_length)} chars")
        with col3:
            st.metric("Features Used", recommender.tfidf_matrix.shape[1])
        
        # Show sample data
        st.subheader("📋 Sample Movies")
        st.dataframe(df[['Movie_Name', 'Storyline']].head(10), use_container_width=True)
        
        # Storyline length distribution
        fig = px.histogram(
            df, 
            x=df['Storyline'].str.len(),
            nbins=30,
            title="Storyline Length Distribution",
            labels={'x': 'Storyline Length (characters)'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🎓 Built as a Data Science Project | 🐍 Python | 🤖 Machine Learning</p>
    </div>
""", unsafe_allow_html=True)