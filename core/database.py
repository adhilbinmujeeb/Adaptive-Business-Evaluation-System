import streamlit as st
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

@st.cache_resource(ttl=3600)
def get_mongo_client():
    for attempt in range(3):
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            return client
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                st.error(f"Failed to connect to MongoDB: {e}")
                st.stop()

def get_database():
    client = get_mongo_client()
    try:
        db = client['business_rag']
        return {
            'question_collection': db['questions'],
            'listings_collection': db['business_listings'],
            'function_cache_collection': db['function_cache'],
            'profile_collection': db['business_profiles'],
            'question_paths_collection': db['question_paths']
        }
    except Exception as e:
        st.error(f"Error accessing MongoDB database: {e}")
        st.stop()
