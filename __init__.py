from flask import Flask
from config import Config

def create_app(config_class=Config):
    """Application Factory Pattern"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Import routes here to avoid circular imports
    from . import routes
    routes.init_routes(app)  # Initialize routes


    return app