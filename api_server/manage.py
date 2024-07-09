import os
from app.__init__ import intialize_app

FLASK_ENV = os.environ.get("FLASK_ENV")
app = intialize_app()

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)