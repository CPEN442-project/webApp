from flask import Flask, request
from pymongo import MongoClient
import os
from dotenv import load_dotenv

"""
This requires following packages to be installed:

os
Flask
"pymongo[srv]"==3.6
dnspython 
python-dotenv


"""
load_dotenv()  # This will load the environment variables from .env file

app = Flask(__name__)

# Use the MONGO_URI from the .env file
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client.llmdetection

def perform_analysis(project_id):
    collection = db.projects # Contains all of the project data
    data = collection.find_one({"title": project_id})

    # Fetch all documents in the collection
    documents = collection.find()

    print("All documents in the collection:")
    print("documents : ", documents)
    # Print each document
    for document in documents:
        print(document)

    # Perform your analysis here
    
    return "Analysis results"

@app.route('/analysis/project_id', methods=['GET'])
def analyze_project():
    project_id = request.args.get('project_id')
    if project_id:
        result = perform_analysis(project_id)
        return result
    else:
        return "Project ID not provided", 400

if __name__ == '__main__':
    perform_analysis("so_actual_work")
    #app.run(port=3005)
