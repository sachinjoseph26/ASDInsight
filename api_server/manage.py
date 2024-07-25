import os
from app.__init__ import intialize_app

FLASK_ENV = os.environ.get("FLASK_ENV")
app = intialize_app()

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '7801'))
    except ValueError:
        PORT = 7801
    app.run(HOST, PORT)