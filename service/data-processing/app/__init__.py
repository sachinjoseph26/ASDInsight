from flask import Flask,redirect
from flask_sqlalchemy import SQLAlchemy
from flask_swagger_ui import get_swaggerui_blueprint

db = SQLAlchemy()

# Flask Initialization
app = Flask(__name__)


# Automatically redirect to Swagger UI
@app.route('/')
def index():
    return redirect('/swagger', code=302)

# Redirect to swagger-api-docs
# @app.route('/')
# def redirect_to_docs():
#     return redirect("http://127.0.0.1:5000/api/docs")

def intialize_app(configName='config'):
    
    swaggerui_blueprint = get_swaggerui_blueprint(
        '/swagger',
        '/static/swagger.json',
        config={
            'app_name': "ASDInsight"
        })
    app.register_blueprint(swaggerui_blueprint)

    app.config.from_object(configName)
    db.init_app(app)

    from app.api import api_bp
    app.register_blueprint(api_bp)

    return app

