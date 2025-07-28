"""
FastAPI application for lung cancer detection
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import base64
from typing import Dict, Any

try:
    from .model import LungCancerModel
    from .preprocessing import preprocess_for_model
    from .utils import SessionTracker, setup_logging
except ImportError:
    from model import LungCancerModel
    from preprocessing import preprocess_for_model
    from utils import SessionTracker, setup_logging

# Setup logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Lung Cancer Detection API",
    description="AI-powered lung cancer detection from chest X-rays",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = LungCancerModel()
session_tracker = SessionTracker()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    await model.load_model()
    logger.info("Model loaded successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model.is_loaded()}

@app.post("/predict")
async def predict_cancer(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict lung cancer from chest X-ray image
    """
    try:
        # Robust file validation
        if not (file.content_type and file.content_type.startswith('image/')) and \
           not (file.filename and file.filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
            raise HTTPException(status_code=400, detail="File must be a valid image (PNG, JPG, JPEG).")

        image_data = await file.read()
        
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted image file: {str(img_error)}")
        
        # Preprocess the image using the dedicated function
        processed_array = preprocess_for_model(image)
        
        # Get prediction from the model
        prediction = await model.predict(processed_array)
        
        # Generate Grad-CAM visualization
        # Pass the preprocessed array for calculation and the original image for overlay
        gradcam_image = await model.generate_gradcam(processed_array, image)
        
        # Convert Grad-CAM to base64
        gradcam_buffer = io.BytesIO()
        gradcam_image.save(gradcam_buffer, format='PNG')
        gradcam_base64 = base64.b64encode(gradcam_buffer.getvalue()).decode()
        
        # Track session
        session_id = session_tracker.create_session(
            prediction=prediction,
            confidence=float(prediction['confidence']),
            filename=file.filename
        )
        
        result = {
            "session_id": session_id,
            "prediction": prediction['class'],
            "confidence": float(prediction['confidence']),
            "details": prediction.get('details', 'No details available.'),
            "probabilities": prediction['probabilities'],
            "gradcam_visualization": f"data:image/png;base64,{gradcam_base64}",
            "timestamp": session_tracker.get_session(session_id)['timestamp']
        }
        
        logger.info(f"Prediction made: {result['prediction']} with confidence {result['confidence']:.3f}")
        return result
        
    except HTTPException as http_exc:
        logger.error(f"HTTP Error: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/sessions")
async def get_sessions():
    """Get all session history"""
    return session_tracker.get_all_sessions()

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get specific session details"""
    session = session_tracker.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
