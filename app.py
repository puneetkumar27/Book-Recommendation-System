import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

# 1. Premium Page Configuration
st.set_page_config(
    page_title="LiteraryLens | Predictive Book Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Custom CSS (Glassmorphism & Micro-interactions)
st.markdown("""
    <style>
    /* Global Background Tuning */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    
    /* Elegant Book Card Design */
    .book-card {
        text-align: center;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 20px 15px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.04);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    /* Smooth Hover Lift Animation */
    .book-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.08);
        border: 1px solid rgba(43, 108, 176, 0.2);
    }
    
    /* Typography Controls */
    .book-title {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #1a202c;
        margin-top: 12px;
        line-height: 1.4;
        min-height: 42px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    /* Decorative Subtle Divider */
    .section-divider {
        margin: 25px 0;
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.1), rgba(0,0,0,0));
    }
    </style>
""", unsafe_allow_html=True)

# 3. Robust Data Loader Block
@st.cache_resource(show_spinner="Optimizing recommendation tensors...")
def load_system_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    try:
        model = pickle.load(open(os.path.join(base_path, 'model.pkl'), 'rb'))
        book_names = pickle.load(open(os.path.join(base_path, 'book_names.pkl'), 'rb'))
        matrix = pickle.load(open(os.path.join(base_path, 'final_matrix.pkl'), 'rb'))
        images_df = pickle.load(open(os.path.join(base_path, 'book_images.pkl'), 'rb'))
    except Exception:
        # Emergency absolute local path fallback
        model = pickle.load(open('model.pkl', 'rb'))
        book_names = pickle.load(open('book_names.pkl', 'rb'))
        matrix = pickle.load(open('final_matrix.pkl', 'rb'))
        images_df = pickle.load(open('book_images.pkl', 'rb'))
        
    return model, book_names, matrix, images_df

model, book_names, matrix, images_df = load_system_data()

# Secure Image URL Parsing Strategy
def get_book_cover(title):
    try:
        url = images_df[images_df['Book-Title'] == title]['Image-URL-M'].values[0]
        if url.startswith('http://'):
            url = url.replace('http://', 'https://')
        return url
    except Exception:
        # High quality geometric placeholder
        return "https://images.unsplash.com/photo-1541963463532-d68292c34b19?w=200&auto=format&fit=crop"

# 4. Premium Sidebar UI Controls
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1506880018603-83d5b814b5a6?w=400&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("🎯 Engine Controls")
    st.markdown("Fine-tune your neural collaborative filtering layer:")
    
    selected_book = st.selectbox(
        "Select an Anchor Book:", 
        options=book_names,
        help="Start typing to look up a book using your pre-compiled vocabulary indices."
    )
    
    # Feature 1: Dynamic Range Limit Control
    num_recommendations = st.slider(
        "Recommendation Limit:", 
        min_value=3, 
        max_value=10, 
        value=5,
        step=1,
        help="Adjust the number of k-Nearest Neighbors calculated by the system."
    )
    
    st.markdown("---")
    search_triggered = st.button('✨ Generate Recommendations', use_container_width=True, type="primary")

# 5. Main Presentation Dashboard
st.title("⚡ LiteraryLens Dashboard")
st.markdown("#### *Advanced Collaborative Filtering Recommendation Engine*")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if search_triggered:
    # Pre-calculate Model Math Metrics
    book_index = np.where(matrix.index == selected_book)[0][0]
    distances, indices = model.kneighbors(
        matrix.iloc[book_index, :].values.reshape(1, -1), 
        n_neighbors=num_recommendations + 1
    )
    
    flat_indices = indices.flatten()
    flat_distances = distances.flatten()
    
    # Feature 2: High-Fidelity Anchor Analytics Header
    with st.container():
        target_col1, target_col2, target_col3, target_col4 = st.columns([1.2, 2.5, 1.5, 1.5])
        
        with target_col1:
            st.image(get_book_cover(selected_book), use_container_width=True)
            
        with target_col2:
            st.subheader("Selected Anchor Item")
            st.markdown(f"### **{selected_book}**")
            st.caption("Now evaluating behavioral reading signatures across the entire user matrix pivot map...")
            
        with target_col3:
            # Calculate raw engagement metrics based on matrix non-zero items
            total_ratings = np.count_nonzero(matrix.loc[selected_book])
            st.metric(label="Global Interaction Count", value=f"{total_ratings} Users")
            
        with target_col4:
            # Average rating of active ratings
            ratings_slice = matrix.loc[selected_book]
            avg_nonzero_rating = ratings_slice[ratings_slice > 0].mean()
            st.metric(label="Calculated Cluster Density", value=f"{avg_nonzero_rating:.2f}/10")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("📚 Top Mathematical Nearest Neighbor Matches")
    
    # Feature 3: Dynamic Grid Framework Generation
    cols = st.columns(num_recommendations)
    
    # Loop seamlessly based on the user's slider preference
    for i in range(1, len(flat_indices)):
        if i > num_recommendations:
            break
            
        rec_title = matrix.index[flat_indices[i]]
        rec_image = get_book_cover(rec_title)
        
        # Calculate algorithmic match percentage using cosine distance vectors
        match_score = (1 - flat_distances[i]) * 100
        
        # Deploy into dynamic columns
        with cols[i - 1]:
            st.markdown(f"""
                <div class="book-card">
                    <div>
                        <img src="{rec_image}" style="height: 200px; object-fit: contain; border-radius: 8px; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.1));">
                        <div class="book-title" title="{rec_title}">{rec_title}</div>
                    </div>
                    <div>
                        <div style="margin-top: 15px; background-color: #e6f4ea; color: #137333; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; display: inline-block;">
                            🔥 {match_score:.1f}% Confidence Match
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
else:
    # Premium Welcome Splash Screen State
    st.empty()
    col_welcome1, col_welcome2 = st.columns([2, 1])
    with col_welcome1:
        st.info("💡 **Getting Started:** Select your anchor book profile in the left control deck and execute the generator. The backend matrix engine handles real-time distance computations based on overlapping reading habits.")
        
        # Display an elegant minimal gallery aesthetic below the tip box
        st.image("https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=1200&auto=format&fit=crop&q=80", use_container_width=True)
    with col_welcome2:
        st.markdown("### Engine Infrastructure")
        st.markdown("""
        * **Distance Metric:** Cosine Similarity
        * **Algorithm Core:** Brute-Force k-Nearest Neighbors (kNN)
        * **Data Matrix Shape:** Sparse User-Item Pivot
        * **Asset Mapping:** Embedded secure Image CDN Links
        """)