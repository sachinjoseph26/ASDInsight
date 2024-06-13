import os

basedir = os.path.abspath(os.path.dirname(__file__))
#get db file
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'persistent/service_1.db')
# set sql settings

