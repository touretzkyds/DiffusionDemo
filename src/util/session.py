"""
Session management for the Diffusion Demo application.

This module provides a SessionManager class to handle user sessions, manage temporary files,
and automatically clean up resources when sessions expire or are no longer active. It implements
a background cleanup thread that periodically checks for and removes old session directories.
"""

import time
import shutil
import threading
from pathlib import Path
from src.util.params import cleanup_interval, cleanup_threshold

class SessionManager:
    """
    Manages user sessions and their associated file directories.
    
    The SessionManager creates unique directories for each user session, tracks
    when sessions were last accessed, and automatically cleans up session data
    that hasn't been accessed for a specified period of time. It implements a
    background cleanup thread to periodically remove expired sessions.
    
    Attributes:
        sessions (dict): Dictionary to store session-specific data
        session_times (dict): Maps session hashes to their last access timestamps
        cleanup_threshold (int): Time in seconds after which inactive sessions are removed
        cleanup_interval (int): Time in seconds between cleanup checks
        _stop_cleanup (bool): Flag to signal the cleanup thread to stop
        _cleanup_thread (Thread): Background thread for periodic cleanup
    """
    def __init__(self):
        """Initialize a new SessionManager instance with default settings."""
        self.sessions = {}  # Store session data
        self.session_times = {}  # Track when sessions were last accessed
        self.cleanup_threshold = cleanup_threshold  # Time in seconds before cleanup
        self.cleanup_interval = cleanup_interval  # Time between cleanup checks
        self._stop_cleanup = False  # Flag to stop the cleanup thread
        self._cleanup_thread = None  # The cleanup thread
    
    def start_cleanup_thread(self):
        """
        Start the background cleanup thread.
        
        The thread will periodically check for and remove session directories
        that haven't been accessed for longer than the cleanup threshold.
        """
        if self._cleanup_thread is None:
            self._stop_cleanup = False
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
    
    def stop_cleanup_thread(self):
        """
        Stop the background cleanup thread.
        
        This method signals the cleanup thread to stop and waits for it to terminate.
        """
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join()
            self._cleanup_thread = None
    
    def _cleanup_loop(self):
        """
        Background loop to periodically cleanup old sessions.
        
        This method runs in a separate thread and calls cleanup_old_sessions()
        at intervals specified by cleanup_interval.
        """
        while not self._stop_cleanup:
            self.cleanup_old_sessions()
            time.sleep(self.cleanup_interval)
    
    def get_session_path(self, session_hash):
        """
        Get the output directory path for a specific session.
        
        Creates the directory if it doesn't exist and updates the last access time
        for the session.
        
        Args:
            session_hash (str): Unique identifier for the user session
            
        Returns:
            Path: Path object pointing to the session directory
        """
        session_dir = Path("outputs") / (session_hash or "default")
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Update the last access time for this session
        self.session_times[session_hash] = time.time()
        return session_dir
    
    def cleanup_session(self, session_hash):
        """
        Clean up session files when the session ends.
        
        Removes the session directory and all its contents, and removes the session
        from the tracking dictionary.
        
        Args:
            session_hash (str): Unique identifier for the user session
        """
        if not session_hash:
            return
            
        session_dir = self.get_session_path(session_hash)
        if session_dir.exists():
            try:
                shutil.rmtree(session_dir)  # Remove the directory and all contents
                if session_hash in self.session_times:
                    del self.session_times[session_hash]  # Remove from tracking
            except Exception as e:
                print(f"Error cleaning up session {session_hash}: {e}")
    
    def cleanup_old_sessions(self):
        """
        Clean up sessions that haven't been accessed for a while.
        
        Checks all tracked sessions and removes those that haven't been accessed
        for longer than the cleanup_threshold.
        """
        current_time = time.time()
        # Use list() to create a copy of items since we'll be modifying the dictionary
        for session_hash, last_access in list(self.session_times.items()):
            if current_time - last_access > self.cleanup_threshold:
                print(f"Cleaning up old session: {session_hash}")
                self.cleanup_session(session_hash)
    
    def get_file_path(self, session_hash, filename):
        """
        Get the full path for a file in the session directory.
        
        Args:
            session_hash (str): Unique identifier for the user session
            filename (str): Name of the file or subdirectory
            
        Returns:
            Path: Path object pointing to the file or subdirectory within the session directory
        """
        return self.get_session_path(session_hash) / filename

# Create a global singleton instance of the SessionManager
session_manager = SessionManager() 