import time
import shutil
import threading
from pathlib import Path
from src.util.params import cleanup_interval, cleanup_threshold

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_times = {}
        self.cleanup_threshold = cleanup_threshold  
        self.cleanup_interval = cleanup_interval  
        self._stop_cleanup = False
        self._cleanup_thread = None
    
    def start_cleanup_thread(self):
        """Start the background cleanup thread"""
        if self._cleanup_thread is None:
            self._stop_cleanup = False
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
    
    def stop_cleanup_thread(self):
        """Stop the background cleanup thread"""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join()
            self._cleanup_thread = None
    
    def _cleanup_loop(self):
        """Background loop to periodically cleanup old sessions"""
        while not self._stop_cleanup:
            self.cleanup_old_sessions()
            time.sleep(self.cleanup_interval)
    
    def get_session_path(self, session_hash):
        """Get the output directory path for a specific session"""
        session_dir = Path("outputs") / (session_hash or "default")
        session_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_times[session_hash] = time.time()
        return session_dir
    
    def cleanup_session(self, session_hash):
        """Clean up session files when the session ends"""
        if not session_hash:
            return
            
        session_dir = self.get_session_path(session_hash)
        if session_dir.exists():
            try:
                shutil.rmtree(session_dir)
                if session_hash in self.session_times:
                    del self.session_times[session_hash]
            except Exception as e:
                print(f"Error cleaning up session {session_hash}: {e}")
    
    def cleanup_old_sessions(self):
        """Clean up sessions that haven't been accessed for a while"""
        current_time = time.time()
        for session_hash, last_access in list(self.session_times.items()):
            if current_time - last_access > self.cleanup_threshold:
                print(f"Cleaning up old session: {session_hash}")
                self.cleanup_session(session_hash)
    
    def get_file_path(self, session_hash, filename):
        """Get the full path for a file in the session directory"""
        return self.get_session_path(session_hash) / filename

session_manager = SessionManager() 