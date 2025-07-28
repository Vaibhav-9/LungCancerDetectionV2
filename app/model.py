"""
Model loading, inference, and Grad-CAM implementation using TorchXRayVision.
This module expects a pre-processed numpy array as input.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import logging
import torchxrayvision as xrv
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LungCancerModel:
    """Lung cancer detection model with Grad-CAM visualization"""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = []
        self.gradcam = None

    async def load_model(self):
        """Load the pre-trained chest X-ray model"""
        try:
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.classes = self.model.pathologies
            self.model.to(self.device)
            self.model.eval()
            self.gradcam = GradCAM(self.model, target_layer=self.model.features, device=self.device)
            logger.info("TorchXRayVision model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    async def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Make prediction on a preprocessed numpy array."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        try:
            # Add batch dimension: (1, 224, 224) -> (1, 1, 224, 224)
            tensor_input = torch.from_numpy(image_array).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_input)
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

            pathology_probs = dict(zip(self.classes, probabilities))
            return self._interpret_probabilities(pathology_probs)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _interpret_probabilities(self, pathology_probs: dict) -> dict:
        """
        Binary cancer classification: Normal vs Potential Cancer
        """
        CANCER_FINDINGS = ['Nodule', 'Mass', 'Lung Lesion']
        
        # Calculate cancer score (max probability among cancer findings)
        cancer_scores = {p: pathology_probs.get(p, 0) for p in CANCER_FINDINGS}
        cancer_score = max(cancer_scores.values()) if cancer_scores else 0.0
        
        # Binary classification with optimized threshold
        if cancer_score >= 0.60:  # Tuned threshold for cancer detection
            predicted_class = "Potential Cancer"
            confidence = cancer_score
            top_finding = max(cancer_scores, key=cancer_scores.get)
            details = f"Cancer indicator: {top_finding} ({cancer_score:.3f})"
        else:
            predicted_class = "Normal"
            confidence = 1.0 - cancer_score
            details = "No significant cancer indicators detected"
            
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'details': details,
            'probabilities': {k: float(v) for k, v in pathology_probs.items()},
            'cancer_indicators': {k: float(v) for k, v in cancer_scores.items()}
        }
    async def generate_gradcam(self, image_array: np.ndarray, original_image: Image.Image) -> Image.Image:
        """Generate Grad-CAM visualization."""
        if not self.is_loaded() or self.gradcam is None:
            raise RuntimeError("Model or Grad-CAM not ready")
        try:
            tensor_input = torch.from_numpy(image_array).unsqueeze(0).to(self.device)
            
            outputs = self.model(tensor_input)
            target_class_index = torch.argmax(outputs, dim=1).item()
            
            cam = self.gradcam.generate_cam(tensor_input, target_class=target_class_index)
            
            original_resized = original_image.resize((224, 224)).convert("RGB")
            return self._create_heatmap_overlay(original_resized, cam)
        except Exception as e:
            logger.error(f"Grad-CAM generation error: {str(e)}")
            return original_image.resize((224, 224)).convert("RGB")

    def _create_heatmap_overlay(self, original_image: Image.Image, cam: np.ndarray) -> Image.Image:
        """Create heatmap overlay on original image."""
        img_array = np.array(original_image)
        cam_normalized = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(overlay)

class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)
        
        if self.gradients is None: return np.zeros(input_tensor.shape[2:])
        
        weights = torch.mean(self.gradients[0], dim=(1, 2))
        cam = torch.zeros(self.activations.shape[2:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0][i]
        
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()
