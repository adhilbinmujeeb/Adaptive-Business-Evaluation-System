import streamlit as st
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

@st.cache_resource(ttl=3600)
def get_mongo_client():
    if not MONGO_URI:
        st.error("MONGO_URI not found in environment variables. Please check your .env file.")
        logger.error("MONGO_URI is missing")
        st.stop()
    for attempt in range(3):
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')  # Test connection
            logger.info("Successfully connected to MongoDB")
            return client
        except Exception as e:
            logger.error(f"MongoDB connection attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                st.error(f"Failed to connect to MongoDB after 3 attempts: {e}")
                st.stop()

def get_database():
    client = get_mongo_client()
    try:
        db = client['business_rag']
        collections = db.list_collection_names()
        expected_collections = [
            'business_attributes',
            'business_listings',
            'questions'
        ]
        for collection in expected_collections:
            if collection not in collections:
                logger.warning(f"Collection '{collection}' not found in database")
                st.warning(f"Collection '{collection}' not found. Some features may be limited.")
        return {
            'business_attributes': db['business_attributes'],
            'business_listings': db['business_listings'],
            'questions': db['questions']
        }
    except Exception as e:
        logger.error(f"Error accessing MongoDB database: {e}")
        st.error(f"Error accessing MongoDB database: {e}")
        st.stop()
