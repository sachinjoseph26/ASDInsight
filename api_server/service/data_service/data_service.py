import sqlite3
from pymongo import MongoClient
from bson import json_util

class DataService:
    def __init__(self, config, mongo):
        self.config = config
        self.mongo = mongo
        self.client = MongoClient(config["MONGO_URI"])
        self.db = self.client['ASD']

    def set_collection(self, collection_name):
        self.collection = self.db[collection_name]

    def fetch_data(self, collection_name, query=None, projection=None):
        self.set_collection(collection_name)
        query = query if query else {}
        projection = projection if projection else {'_id': 0}
        data_cursor = self.collection.find(query, projection)
        data_list = list(data_cursor)
        json_data = json_util.dumps(data_list)
        return json_data
    
    def insert_data(self, collection_name, data_list):
        self.set_collection(collection_name)
        self.collection.insert_many(data_list)
        return f'{len(data_list)} records inserted into MongoDB collection {self.collection.name}'


    def close_connection(self):
        self.client.close()