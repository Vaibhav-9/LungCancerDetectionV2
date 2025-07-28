"""
Streamlit UI for Lung Cancer Detection
"""
import streamlit as st
import requests
import json
from PIL import Image
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-card {
        background-color: #f0f2f6;
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-card h3 {
        margin-top: 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 0.5rem 0;
    }
    .sidebar-info {
        background-color: #e8f4f8;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border: 1px solid #bdc3c7;
    }
    .sidebar-info h4 {
        color: #1f77b4;
        margin-top: 0;
    }
    .sidebar-info ul {
        color: #34495e;
        margin-bottom: 0;
    }
    .sidebar-info li {
        margin-bottom: 0.3rem;
    }
    
    /* Ensure good contrast for all info boxes */
    .stInfo {
        background-color: #e8f4f8 !important;
        color: #2c3e50 !important;
    }
    
    /* Style for warning boxes */
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
    }
    
    /* Style for success boxes */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_and_predict(uploaded_file):
    """Upload file and get prediction from API"""
    files = {"file": uploaded_file}
    
    with st.spinner("Analyzing chest X-ray..."):
        try:
            response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                # Client error (bad request)
                try:
                    error_detail = response.json().get('detail', 'Bad request')
                    st.error(f"Upload Error: {error_detail}")
                except:
                    st.error("Invalid file format. Please upload a valid image file.")
                return None
            else:
                st.error(f"API Error: {response.status_code}")
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"Details: {error_detail}")
                except:
                    pass
                return None
        except requests.exceptions.Timeout:
            st.error("Request timed out. The image might be too large or the server is busy.")
            return None
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API server. Please make sure it's running.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {str(e)}")
            return None

def get_session_history():
    """Get session history from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/sessions", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def display_prediction_results(result):
    """Display prediction results in a nice format"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔍 Prediction Results")
        
        # Main prediction
        prediction = result['prediction']
        confidence = result['confidence']
        
        # Color coding based on prediction
        color_map = {
            'Normal': 'green',
            'Potential Nodule/Mass': 'red',
            'Other Finding': 'orange'
        }
        
        color = color_map.get(prediction, 'blue')
        
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="color: {color}; margin: 0;">Prediction: {prediction}</h3>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                Confidence: <strong>{confidence:.1%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Grad-CAM Visualization")
        
        # Display Grad-CAM visualization
        if 'gradcam_visualization' in result:
            gradcam_data = result['gradcam_visualization']
            if gradcam_data.startswith('data:image'):
                # Extract base64 data
                base64_data = gradcam_data.split(',')[1]
                image_data = base64.b64decode(base64_data)
                gradcam_image = Image.open(io.BytesIO(image_data))
                
                st.image(gradcam_image, caption="Areas of interest highlighted", use_container_width=True)
            else:
                st.warning("Grad-CAM visualization not available")
        
        # Session info
        st.markdown("#### Session Information")
        st.info(f"""
        **Session ID**: {result['session_id'][:8]}...
        **Timestamp**: {result['timestamp']}
        """)

def display_statistics_dashboard():
    """Display statistics dashboard"""
    st.markdown("## 📊 Statistics Dashboard")
    
    sessions = get_session_history()
    
    if not sessions:
        st.warning("No session data available.")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(sessions)
    
    # Extract prediction class from nested structure
    df['prediction_class'] = df['prediction'].apply(lambda x: x.get('class', 'Unknown') if isinstance(x, dict) else 'Unknown')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", len(df))
    
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col3:
        feedback_count = df['user_feedback'].notna().sum()
        st.metric("Sessions with Feedback", feedback_count)
    
    with col4:
        latest_session = df['timestamp'].max()
        days_ago = (datetime.now() - latest_session.replace(tzinfo=None)).days
        st.metric("Days Since Last Session", days_ago)
    
    # Prediction distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Prediction Distribution")
        prediction_counts = df['prediction_class'].value_counts()
        
        fig = px.pie(
            values=prediction_counts.values,
            names=prediction_counts.index,
            title="Distribution of Predictions"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Confidence Distribution")
        
        fig = px.histogram(
            df,
            x='confidence',
            nbins=20,
            title="Distribution of Confidence Scores"
        )
        fig.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline
    st.markdown("### Sessions Over Time")
    df_daily = df.groupby(df['timestamp'].dt.date).size().reset_index()
    df_daily.columns = ['date', 'count']
    
    fig = px.line(
        df_daily,
        x='date',
        y='count',
        title="Daily Session Count"
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🫁 Lung Cancer Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Controls")
        
        # API Health Check
        st.markdown("### API Status")
        if check_api_health():
            st.success("✅ API is running")
        else:
            st.error("❌ API is not accessible")
            st.warning("Please make sure the FastAPI server is running on localhost:8000")
        
        st.markdown("### Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📤 Upload & Analyze", "📊 Statistics", "ℹ️ About"]
        )
        
        # Info section
        st.markdown("""
        <div class="sidebar-info">
            <h4>💡 Tips</h4>
            <ul>
                <li>Upload clear chest X-ray images</li>
                <li>Supported formats: JPG, PNG, JPEG</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content based on page selection
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Lung Cancer Detection System
        
        This AI-powered system analyzes chest X-ray images to detect potential signs of lung cancer.
        The system uses deep learning models trained on medical imaging data to provide predictions
        with confidence scores and visual explanations through Grad-CAM technology.
        
        ### 🎯 Features:
        - **AI-Powered Analysis**: Advanced deep learning models for accurate predictions
        - **Grad-CAM Visualization**: See which areas the AI focuses on
        - **Confidence Scoring**: Get reliability scores for each prediction
        - **Session Tracking**: Keep track of all your analyses
        - **Statistics Dashboard**: View insights from all predictions
        """)
        
    elif page == "📤 Upload & Analyze":
        st.markdown("## 📤 Upload & Analyze Chest X-Ray")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                
                # Image info
                st.markdown("#### Image Information")
                st.info(f"""
                **Filename**: {uploaded_file.name}
                **Size**: {image.size[0]} x {image.size[1]} pixels
                **Format**: {image.format}
                **Mode**: {image.mode}
                """)
            
            with col2:
                if st.button("🔍 Analyze Image", type="primary"):
                    if check_api_health():
                        # Validate file before sending
                        try:
                            # Test if we can open the image
                            test_image = Image.open(uploaded_file)
                            test_image.verify()
                            
                            # Reset file pointer after verification
                            uploaded_file.seek(0)
                            
                            # Get prediction
                            result = upload_and_predict(uploaded_file)
                            
                            if result:
                                st.session_state['last_result'] = result
                                st.success("✅ Analysis completed!")
                            else:
                                st.error("❌ Analysis failed. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Invalid image file: {str(e)}")
                            st.info("Please upload a valid image file (JPG, PNG, JPEG)")
                    else:
                        st.error("API is not available. Please check the server.")
        
        # Display results if available
        if 'last_result' in st.session_state:
            st.markdown("---")
            display_prediction_results(st.session_state['last_result'])
            
            # Feedback section
            st.markdown("### 💬 Feedback")
            feedback = st.text_area("Please provide your feedback on this prediction (optional):")
            
            if st.button("Submit Feedback"):
                if feedback:
                    st.success("Thank you for your feedback!")
                    # Here you could send feedback to the API
                else:
                    st.warning("Please enter some feedback first.")
    
    elif page == "📊 Statistics":
        display_statistics_dashboard()
        
    elif page == "ℹ️ About":
        st.markdown("""
        ## ℹ️ About This System
        
        ### 🧠 Technology Stack:
        - **Backend**: FastAPI for high-performance API
        - **Frontend**: Streamlit for interactive web interface
        - **AI Model**: ResNet50 deep learning architecture
        - **Visualization**: Grad-CAM for explainable AI
        - **Image Processing**: OpenCV and PIL for preprocessing
        
        ### 🎯 Model Information:
        - **Architecture**: Convolutional Neural Network (CNN)
        - **Input**: 224x224 RGB chest X-ray images
        - **Output**: 3 classes (Normal, Benign, Malignant)
        - **Features**: Confidence scoring and visual explanations
        """)

if __name__ == "__main__":
    main()
