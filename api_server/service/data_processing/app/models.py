from api_server.app import db

class Model1(db.Model):
    __tablename__ = 'table_name'
    id = db.Column(db.Integer, primary_key=True)