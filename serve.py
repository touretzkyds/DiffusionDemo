from flask import Flask, send_file
from src.pipelines.embeddings import get_user_dir, get_user_examples_dir

flask_app = Flask(__name__)


@flask_app.route("/plot/<session_hash>")
def serve_user_plot(session_hash):
    user_dir = get_user_dir(session_hash)
    if not user_dir:
        return "Invalid session", 404

    html_path = user_dir / "embedding_plot.html"
    abs_html_path = html_path.absolute()
    str_html_path = str(abs_html_path)

    print(f"Trying to serve file at: {str_html_path}")
    print(f"File exists: {html_path.exists()}")
    print(f"Absolute path: {abs_html_path}")

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
    examples_dir = get_user_examples_dir(session_hash)
    if not examples_dir:
        return "Invalid session", 404

    image_path = examples_dir / image_name
    abs_image_path = image_path.absolute()
    str_image_path = str(abs_image_path)

    print(f"Trying to serve image at: {str_image_path}")
    print(f"Image exists: {image_path.exists()}")

    if image_path.exists():
        try:
            return send_file(str_image_path, mimetype="image/jpeg")
        except Exception as e:
            print(f"Error serving user example image: {e}")
            return f"Error serving image: {e}", 500
    else:
        return f"Image not found at {str_image_path}", 404


def run_flask_server():
    try:
        print("Starting Flask server on port 8050")
        flask_app.run(host="0.0.0.0", port=8050, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error starting Flask server: {e}")
