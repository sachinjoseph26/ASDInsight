import os

class Config:
    basedir = os.path.abspath(os.path.dirname(__file__))
    DB_SQL_PATH = 'sqlite:///' + os.path.join(basedir, 'persistent/service_1.db')
    DB_MONGO_PATH = "mongodb+srv://sachinjoseph054:kOpRxNfjcc1GC74w@asdcluster.id1l6xq.mongodb.net/ASD"

