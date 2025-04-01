"""
Flask Server for Auxiliary Content in Diffusion Demo

This module implements a Flask server that runs alongside the main Gradio application
to serve content that Gradio can't easily handle directly, such as interactive plots
and dynamically generated images. It provides endpoints for:
1. Serving interactive 3D embedding plots created with Plotly
2. Serving user-specific example images stored in session directories

The server runs on port 8050 and creates a public tunnel URL to make the content 
accessible from external networks.
"""

from flask import Flask, send_file
from src.util.base import get_user_dir, get_user_examples_dir
from gradio.networking import setup_tunnel
import secrets

# Initialize the Flask application
flask_app = Flask(__name__)

# Global variable to store the tunnel URL for accessing the Flask server externally
flask_tunnel = None


@flask_app.route("/plot/<session_hash>")
def serve_user_plot(session_hash):
    """
    Serve a user's interactive embedding plot HTML file.
    
    This endpoint retrieves and serves the 3D visualization of word embeddings
    that was generated for a specific user session. The plot is an HTML file
    created with Plotly that contains JavaScript for interactive visualization.
    
    Args:
        session_hash (str): Unique identifier for the user session
        
    Returns:
        HTML file response or error message
    """
    # Get the directory for this user's session
    user_dir = get_user_dir(session_hash)
    if not user_dir:
        return "Invalid session", 404

    # Construct the path to the HTML plot file
    html_path = user_dir / "embedding_plot.html"
    abs_html_path = html_path.absolute()
    str_html_path = str(abs_html_path)

    # Debugging information
    print(f"Trying to serve file at: {str_html_path}")
    print(f"File exists: {html_path.exists()}")
    print(f"Absolute path: {abs_html_path}")

    # Serve the file if it exists, otherwise return an error
    if html_path.exists():
        try:
            return send_file(str_html_path, mimetype="text/html")
        except Exception as e:
            print(f"Error serving file: {e}")
            return f"Error serving file: {e}", 500
    else:
        return f"Plot not found at {str_html_path}", 404


@flask_app.route("/examples/<session_hash>/<image_name>")
def serve_user_example_image(session_hash, image_name):
    """
    Serve an example image from a user's session directory.
    
    This endpoint retrieves and serves images that were generated for specific examples
    in a user session. These images are typically used in the embedding visualization
    to show generated content for specific words.
    
    Args:
        session_hash (str): Unique identifier for the user session
        image_name (str): Name of the image file to serve
        
    Returns:
        Image file response or error message
    """
    # Get the examples directory for this user's session
    examples_dir = get_user_examples_dir(session_hash)
    if not examples_dir:
        return "Invalid session", 404

    # Construct the path to the image file
    image_path = examples_dir / image_name
    abs_image_path = image_path.absolute()
    str_image_path = str(abs_image_path)

    # Debugging information
    print(f"Trying to serve image at: {str_image_path}")
    print(f"Image exists: {image_path.exists()}")

    # Serve the image if it exists, otherwise return an error
    if image_path.exists():
        try:
            return send_file(str_image_path, mimetype="image/jpeg")
        except Exception as e:
            print(f"Error serving user example image: {e}")
            return f"Error serving image: {e}", 500
    else:
        return f"Image not found at {str_image_path}", 404


def run_flask_server():
    """
    Start the Flask server with a public tunnel.
    
    This function initializes and starts the Flask server on port 8050 and creates
    a public tunnel URL using Gradio's tunneling feature. This allows the content
    to be accessible from external networks even if the server is behind a firewall.
    
    The function sets the global flask_tunnel variable to store the public URL.
    """
    global flask_tunnel
    try:
        # Set up a tunnel for the Flask server with a random token for security
        print("Setting up Flask tunnel on port 8050")
        flask_tunnel = setup_tunnel("0.0.0.0", 8050, secrets.token_urlsafe(32), None, None)
        print(f"Flask tunnel URL: {flask_tunnel}")
        
        # Start the Flask server on port 8050, accessible from all network interfaces
        print("Starting Flask server on port 8050")
        flask_app.run(host="0.0.0.0", port=8050, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error starting Flask server: {e}")


def get_flask_tunnel_url():
    """
    Get the public URL for the Flask tunnel.
    
    Returns:
        str: The public URL where the Flask server can be accessed
    """
    return flask_tunnel