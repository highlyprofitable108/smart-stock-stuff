import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Common method to get MongoDB URI
def get_mongo_uri():
    return os.getenv('MONGO_URI')


# Common method to connect to MongoDB
def connect_to_mongodb():
    mongo_client = MongoClient(get_mongo_uri())
    db = mongo_client.stock_data  # Change 'stock_data' to your actual database name
    return db


# Common method to get the full schema for a MongoDB database
def get_mongodb_schema(db):
    collections = db.list_collection_names()
    schema = {}

    for collection_name in collections:
        collection = db.get_collection(collection_name)
        documents = collection.find().limit(1000)
        collection_schema = {}

        for document in documents:
            for key in document.keys():
                if key not in collection_schema:
                    collection_schema[key] = type(document[key]).__name__

        schema[collection_name] = collection_schema

    return schema


# Main function
def main():
    # Connect to MongoDB
    db = connect_to_mongodb()

    # Get MongoDB schema
    schema = get_mongodb_schema(db)

    # Print schema
    print(schema)


if __name__ == "__main__":
    main()
