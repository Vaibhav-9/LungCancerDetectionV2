"""
Session tracking and logging utilities
"""
import uuid
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

def setup_logging(log_level: str = "INFO", log_file: str = "lung_cancer_detection.log") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file name
    
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")
    
    return logger

class SessionTracker:
    """Track user sessions and predictions"""
    
    def __init__(self, session_file: str = "sessions.json"):
        self.session_file = Path("data") / session_file
        self.sessions = {}
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        self.session_file.parent.mkdir(exist_ok=True)
        
        # Load existing sessions
        self._load_sessions()
    
    def _load_sessions(self):
        """Load sessions from file"""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    self.sessions = json.load(f)
                self.logger.info(f"Loaded {len(self.sessions)} sessions")
            else:
                self.sessions = {}
                self.logger.info("No existing sessions found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading sessions: {str(e)}")
            self.sessions = {}
    
    def _save_sessions(self):
        """Save sessions to file"""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.sessions, f, indent=2, default=str)
            self.logger.debug("Sessions saved to file")
        except Exception as e:
            self.logger.error(f"Error saving sessions: {str(e)}")
    
    def create_session(self, prediction: Dict[str, Any], 
                      confidence: float, filename: str) -> str:
        """
        Create a new session
        
        Args:
            prediction: Model prediction result
            confidence: Prediction confidence score
            filename: Original filename
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'prediction': prediction,
            'confidence': confidence,
            'user_feedback': None,
            'notes': None
        }
        
        self.sessions[session_id] = session_data
        self._save_sessions()
        
        self.logger.info(f"Created session {session_id} for file {filename}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session by ID
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data or None if not found
        """
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all sessions
        
        Returns:
            List of all session data
        """
        return list(self.sessions.values())
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session data
        
        Args:
            session_id: Session identifier
            updates: Dictionary with updates
        
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.sessions:
            self.logger.warning(f"Session {session_id} not found")
            return False
        
        self.sessions[session_id].update(updates)
        self._save_sessions()
        
        self.logger.info(f"Updated session {session_id}")
        return True
    
    def add_feedback(self, session_id: str, feedback: str, notes: str = None) -> bool:
        """
        Add user feedback to session
        
        Args:
            session_id: Session identifier
            feedback: User feedback
            notes: Additional notes
        
        Returns:
            True if successful, False otherwise
        """
        updates = {
            'user_feedback': feedback,
            'feedback_timestamp': datetime.now().isoformat()
        }
        
        if notes:
            updates['notes'] = notes
        
        return self.update_session(session_id, updates)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get session statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.sessions:
            return {
                'total_sessions': 0,
                'predictions_by_class': {},
                'average_confidence': 0,
                'sessions_with_feedback': 0
            }
        
        sessions_list = list(self.sessions.values())
        
        # Count predictions by class
        predictions_count = {}
        confidences = []
        feedback_count = 0
        
        for session in sessions_list:
            pred_class = session['prediction'].get('class', 'Unknown')
            predictions_count[pred_class] = predictions_count.get(pred_class, 0) + 1
            
            confidences.append(session['confidence'])
            
            if session.get('user_feedback'):
                feedback_count += 1
        
        stats = {
            'total_sessions': len(sessions_list),
            'predictions_by_class': predictions_count,
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'sessions_with_feedback': feedback_count,
            'feedback_rate': feedback_count / len(sessions_list) if sessions_list else 0
        }
        
        return stats
    
    def export_sessions(self, export_file: str = None) -> str:
        """
        Export sessions to CSV file
        
        Args:
            export_file: Export filename
        
        Returns:
            Path to exported file
        """
        if not export_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = f"sessions_export_{timestamp}.csv"
        
        export_path = Path("exports") / export_file
        export_path.parent.mkdir(exist_ok=True)
        
        try:
            import pandas as pd
            
            # Convert sessions to DataFrame
            sessions_data = []
            for session in self.sessions.values():
                row = {
                    'session_id': session['session_id'],
                    'timestamp': session['timestamp'],
                    'filename': session['filename'],
                    'prediction_class': session['prediction'].get('class', 'Unknown'),
                    'confidence': session['confidence'],
                    'user_feedback': session.get('user_feedback'),
                    'notes': session.get('notes')
                }
                sessions_data.append(row)
            
            df = pd.DataFrame(sessions_data)
            df.to_csv(export_path, index=False)
            
            self.logger.info(f"Sessions exported to {export_path}")
            return str(export_path)
            
        except ImportError:
            self.logger.warning("pandas not available, using basic CSV export")
            
            # Basic CSV export without pandas
            import csv
            
            with open(export_path, 'w', newline='') as csvfile:
                fieldnames = ['session_id', 'timestamp', 'filename', 'prediction_class', 
                             'confidence', 'user_feedback', 'notes']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for session in self.sessions.values():
                    row = {
                        'session_id': session['session_id'],
                        'timestamp': session['timestamp'],
                        'filename': session['filename'],
                        'prediction_class': session['prediction'].get('class', 'Unknown'),
                        'confidence': session['confidence'],
                        'user_feedback': session.get('user_feedback', ''),
                        'notes': session.get('notes', '')
                    }
                    writer.writerow(row)
            
            self.logger.info(f"Sessions exported to {export_path}")
            return str(export_path)

def create_directories():
    """Create necessary directories for the application"""
    directories = ['logs', 'data', 'exports', 'models', 'uploads']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logging.getLogger(__name__).info("Created application directories")

def get_system_info() -> Dict[str, Any]:
    """
    Get system information
    
    Returns:
        Dictionary with system information
    """
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
    }
    
    # Try to get additional system info if psutil is available
    try:
        import psutil
        info.update({
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent
        })
    except ImportError:
        # Fallback without psutil
        info.update({
            'memory_total': 'Not available (psutil not installed)',
            'memory_available': 'Not available (psutil not installed)',
            'disk_usage': 'Not available (psutil not installed)'
        })
    
    return info
