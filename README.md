# Lung Cancer Detection System

An AI-powered web application for detecting potential lung abnormalities from chest X-ray images. This tool uses a deep learning model to provide predictions and visual explanations.

-----

## Features

  - **AI-Powered Analysis**: Utilizes a `DenseNet-121` model pre-trained on chest X-rays for analyzing images.
  - **Explainable AI**: Generates a `Grad-CAM` visualization to highlight the areas the model focused on for its prediction.
  - **Interactive Web App**: A user-friendly interface built with Streamlit for easy image uploading and analysis.
  - **FastAPI Backend**: The system is powered by a high-performance REST API.
  - **Session History & Stats**: Tracks all predictions and provides a dashboard to view analytics.

-----

## Technology Stack

  - **Backend**: FastAPI
  - **Frontend**: Streamlit
  - **AI Model**: PyTorch, TorchXRayVision
  - **Image Processing**: OpenCV, Pillow
  - **Data Handling**: Pandas, NumPy

-----

## Getting Started

### Prerequisites

  - Python 3.8+
  - `pip` package manager

### Installation and Running the App

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd lung_cancer_detection
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Start the FastAPI backend server:**

    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

    The API will be live at `http://localhost:8000`.

5.  **Open a new terminal** and start the Streamlit frontend:

    ```bash
    streamlit run streamlit_app.py
    ```

    The web application will open in your browser at `http://localhost:8501`.

-----

## How to Use

### Web Interface

1.  Navigate to the **Upload & Analyze** page in the web app.
2.  Upload a chest X-ray image (`.png`, `.jpg`, `.jpeg`).
3.  Click the **Analyze Image** button to get the prediction.
4.  Review the prediction, confidence score, and the Grad-CAM heat map.

### API Usage

You can also interact with the API directly.

  - **Health Check**: `GET /health`
  - **Get All Sessions**: `GET /sessions`
  - **Predict from an image file**: `POST /predict`

**Example using Python's `requests` library:**

```python
import requests

file_path = 'path/to/your/chest_xray.jpg'

with open(file_path, 'rb') as f:
    files = {'file': (f.name, f, 'image/jpeg')}
    response = requests.post('http://localhost:8000/predict', files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities available for {len(result['probabilities'])} findings.")
    else:
        print(f"Error: {response.text}")
```

**API Response Format:**
The API returns a JSON object with detailed results from the model.

```json
{
  "session_id": "a1b2c3d4-...",
  "prediction": "Potential Nodule/Mass",
  "confidence": 0.85,
  "probabilities": {
    "Nodule": 0.85,
    "Mass": 0.75,
    "Pneumonia": 0.15,
    "...": "..."
  },
  "gradcam_visualization": "data:image/png;base64,..."
}
```
