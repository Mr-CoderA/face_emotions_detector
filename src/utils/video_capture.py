"""
Video capture utility for capturing frames from webcam or video file
"""
import cv2
import time
import threading
import queue
from typing import Optional, Tuple, Union
import numpy as np

import src.config as config

class VideoCaptureThread:
    def __init__(self, source=0, width=None, height=None, buffer_size=5):
        """
        Initialize video capture with threaded frame reading.
        
        Args:
            source: Camera index, path to video file, or RTSP URL
            width: Target width for captured frames or None for default
            height: Target height for captured frames or None for default
            buffer_size: Maximum number of frames to buffer
        """
        self.source = source
        self.width = width or config.DISPLAY_WIDTH
        self.height = height or config.DISPLAY_HEIGHT
        self.buffer_size = buffer_size
        
        # Create frame buffer
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Initialize capture
        self.cap = cv2.VideoCapture(source)
        
        # Set resolution if specified
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Check if capture is opened successfully
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
            
        # Get actual resolution
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video capture initialized with resolution: {self.actual_width}x{self.actual_height}")
        
        # Thread control
        self.running = False
        self.thread = None
        
        # Start capture thread
        self.start()
    
    def start(self):
        """Start the capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        """Main capture loop running in a separate thread"""
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                
                if not ret:
                    # Failed to get frame, might be end of video or disconnection
                    print("Failed to get frame from source. Retrying...")
                    time.sleep(0.5)  # Wait before retry
                    continue
                
                # Resize frame if it doesn't match target dimensions
                if (self.width and self.height and 
                    (frame.shape[1] != self.width or frame.shape[0] != self.height)):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Add timestamp to the frame
                timestamp = time.time()
                
                # Put frame in queue
                self.frame_queue.put((frame, timestamp))
            else:
                # Queue is full, sleep briefly
                time.sleep(0.01)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the capture.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.frame_queue.empty():
            # No frames available yet
            return False, None
            
        # Get frame from queue
        frame, _ = self.frame_queue.get()
        return True, frame
    
    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the most recent frame, skipping older frames.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.frame_queue.empty():
            # No frames available
            return False, None
            
        # Get the most recent frame by emptying the queue
        frame = None
        timestamp = 0
        
        while not self.frame_queue.empty():
            try:
                frame, timestamp = self.frame_queue.get(block=False)
            except queue.Empty:
                break
                
        if frame is None:
            return False, None
            
        return True, frame
    
    def stop(self):
        """Stop the capture thread and release resources"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.stop() 