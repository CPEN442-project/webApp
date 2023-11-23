from flask import Flask, request
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datetime import datetime
from collections import Counter
from nltk.util import ngrams
import re
import statistics


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Use the MONGO_URI from the .env file
mongo_client = MongoClient(os.getenv('MONGO_URI'))


# Function to calculate the time difference between events
def time_difference(event1, event2):
    time_format = "%Y-%m-%dT%H:%M:%S.%fZ"  # Adjust this format to match your timestamp format
    return (datetime.strptime(event2['timestamp'], time_format) - 
            datetime.strptime(event1['timestamp'], time_format)).total_seconds()

# Function to check database connection
def check_db_connection(client):
    try:
        # This command forces a round trip to the server
        client.admin.command('ping')
        print("MongoDB connection successful.")
        return True
    except Exception as e:
        print("Error connecting to MongoDB: ", e)
        return False

# Function to check if database and collection exist
def check_db_and_collection(client, db_name, collection_name):
    if db_name in client.list_database_names():
        db = client[db_name]
        if collection_name in db.list_collection_names():
            print(f"Database '{db_name}' and collection '{collection_name}' exist.")
            return True
        else:
            print(f"Collection '{collection_name}' does not exist in database '{db_name}'.")
            return False
    else:
        print(f"Database '{db_name}' does not exist.")
        return False
    

def analyze_text_ngrams(n,textState):
    # Tokenize the text by words
    words = re.findall(r'\w+', textState)
    # Generate trigrams
    ngrams = list(ngrams(words, n))

    # Count the occurrences of each trigram
    ngram_counts = Counter(ngrams)

    # Get the most common trigrams
    most_common_ngrams = ngram_counts.most_common(10)  # Adjust the number as needed

    return most_common_ngrams


def calculate_cps_for_segments(segments):
    cps_list = []

    for segment in segments:
        segment_duration_seconds = sum(time_difference(segment[i], segment[i + 1]) for i in range(len(segment) - 1))
        character_count = sum(1 for event in segment if event['type'] == 'keyboard')
        # Calculate CPS for the segment
        cps = (character_count / segment_duration_seconds) if segment_duration_seconds > 0 else 0
        cps_list.append(cps)

    return cps_list


def detect_significant_cps_change(segments, threshold_multiplier=2):
    cps_list = calculate_cps_for_segments(segments)
    significant_changes_count = 0
    total_cps_difference = 0

    for i in range(1, len(cps_list)):
        cps_difference = abs(cps_list[i] - cps_list[i - 1])
        if cps_difference > threshold_multiplier * statistics.median(cps_list):
            significant_changes_count += 1
            total_cps_difference += cps_difference

    average_cps_difference = total_cps_difference / significant_changes_count if significant_changes_count > 0 else 0

    return significant_changes_count, average_cps_difference


def segment_events(events, time_diffs):
    # Calculate the median of time differences
    median_time_diff = statistics.median(time_diffs)

    # Initialize segments
    segments = []
    current_segment = [events[0]]

    for i in range(1, len(events)):
        if time_diffs[i-1] > 5 * median_time_diff:
            # Start a new segment
            segments.append(current_segment)
            current_segment = [events[i]]
        else:
            # Continue the current segment
            current_segment.append(events[i])

    # Add the last segment
    if current_segment:
        segments.append(current_segment)

    return segments

def create_user_metrics(events, textState):
    # Initialize counters
    backspace_count = 0
    total_keystrokes = 0
    consecutive_backspaces = 0
    backspace_sequences = []

    # Process events
    for event in events:
        if event['type'] == 'keyboard':
            total_keystrokes += 1

            # Check for backspace/delete
            if event['data'] in ['Backspace', 'Delete']:
                backspace_count += 1
                consecutive_backspaces += 1
            else:
                if consecutive_backspaces > 0:
                    backspace_sequences.append(consecutive_backspaces)
                    consecutive_backspaces = 0

    # Add the last sequence if it exists
    if consecutive_backspaces > 0:
        backspace_sequences.append(consecutive_backspaces)

    ###### Calculate error rate and average backspace sequence length ###### 
    error_rate = backspace_count / total_keystrokes if total_keystrokes > 0 else 0
    average_backspace_seq = sum(backspace_sequences) / len(backspace_sequences) if backspace_sequences else 0

    ###### Calculate user wording behaviour ###### 
    n = 2
    most_common_ngrams = analyze_text_ngrams(n, textState)

    
    # Calculate time differences between consecutive events
    time_diffs = [time_difference(events[i], events[i + 1]) for i in range(len(events) - 1)]

    # Segment the events
    keystroke_segments = segment_events(events, time_diffs)


    ###### calculate average cps ######
    total_cps = 0
    segment_count = 0

    for segment in keystroke_segments:
        segment_duration_seconds = sum(time_difference(segment[i], segment[i + 1]) for i in range(len(segment) - 1))
        character_count = sum(1 for event in segment if event['type'] == 'keyboard')
        # Calculate CPS for the segment
        cps = (character_count / segment_duration_seconds) if segment_duration_seconds > 0 else 0
        total_cps += cps
        segment_count += 1

    # Calculate average CPS across all segments ()
    average_cps = total_cps / segment_count if segment_count > 0 else 0

    ###### significant cps changes count, and average cps difference between segments ######
    significant_changes_count, average_cps_difference = detect_significant_cps_change(keystroke_segments)


    return {

        # number of backspaces over total keystrokes
        "error_rate": error_rate,
        # Average number of consecutive backspaces in the events
        "average_consecutive_backspaces": average_backspace_seq,
        # Top 10 common bigrams the user used
        "most_common_ngrams": (n, most_common_ngrams),
        # average character per minute for the entire event array
        "average_cps": average_cps,
        # Number if Calculate significant changes in CPS through out events
        "significant_changes_count": significant_changes_count,
        # Average CPS difference between significant changes
        "average_cps_difference": average_cps_difference,

    }




def create_labelled_training_data(username):
    # Ensure MongoDB connection
    if not check_db_connection(mongo_client):
        raise Exception("Failed to connect to MongoDB.")

    # Database and collection setup
    db_name = 'llmdetection'
    collection_name = 'projects'
    if not check_db_and_collection(mongo_client, db_name, collection_name):
        raise Exception(f"Database '{db_name}' or collection '{collection_name}' does not exist.")

    db = mongo_client[db_name]
    collection = db[collection_name]

    # Fetch projects related to the user
    #user_projects = collection.find({"username": username})
    if username == "so":
        project_names = ["so_actual_work", "so_actual_work2"]
    elif username == "april":
        project_names = ["Aj_genuine", "Aj_genuine2"]
    elif username == "aleks":
        project_names = ["Aleks_genuine"]
    elif username == "roger":
        project_names = ["roger_genuine", "roger_genuine2"]
    elif username == "cheat_behaviour":
        project_names = ["Aleks_fake_1", "Aleks_fake-2", "roger_copypasting", "roger_copypasting2", "Aj_fake_1", "Aj_fake_2","so_fake_1", "so_fake_2"]

    
    if username == "so":
        user_projects = collection.find({"title": {"$in": project_names}})
    elif username == "april":
        user_projects = collection.find({"title": {"$in": project_names}})

    for project in user_projects:
        if 'events' not in project:
            continue  # Skip if there are no events

        # Extract keyboard and mouse events
        keyboard_events = [event for event in project['events'] if event['type'] == 'keyboard']
        mouse_events = [event for event in project['events'] if event['type'] == 'mouse']
        textState = project.get('textState', '')

        # Analyze typing behavior
        user_typing_metrics = user_typing_metrics(keyboard_events, textState)

        # Save the results to MongoDB or process further
        # For example, saving the result:
        # result_collection = db['user_metrics']
        # result_collection.insert_one({
        #     "username": username,
        #     "project_title": project['title'],
        #     "typing_metrics": user_typing_metrics
        # })

@app.route('/analysis', methods=['GET'])
def analyze_project():
    project_id = request.args.get('project_id')
    # http://<hostname>:<port>/analysis?project_id=<value>.
    if project_id:
        result = perform_analysis(project_id)
        return result
    else:
        return "Project ID not provided", 400

if __name__ == '__main__':
    #transfer_labelled_data()


    # username = "so"
    # create_user_metrics(username)
    #app.run(port=3005)






# This is one time code to transfer labelled data from one collection to another
def transfer_labelled_data():
    db_name = 'llmdetection'

    # Source and target collection names
    source_collection_name = 'projects'
    target_collection_name = 'labelled_training_data'

    # Connect to the database and collections
    db = mongo_client[db_name]
    source_collection = db[source_collection_name]
    target_collection = db[target_collection_name]

    # Define the user project names
    user_project_mapping = {
        "so": ["so_actual_work", "so_actual_work2"],
        "april": ["Aj_genuine", "Aj_genuine2"],
        "aleks": ["Aleks_genuine"],
        "roger": ["roger_genuine", "roger_genuine2"],
        "cheat_behaviour": ["Aleks_fake_1", "Aleks_fake-2", "roger_copypasting", "roger_copypasting2", "Aj_fake_1", "Aj_fake_2", "so_fake_1", "so_fake_2"]
    }

    for username, project_names in user_project_mapping.items():
        # Retrieve projects for the current user
        user_projects = source_collection.find({"title": {"$in": project_names}})
        
        # Set the label based on the username
        label = "cheat" if username == "cheat_behaviour" else "genuine"
        
        # Duplicate each project into the target collection with the added label
        for project in user_projects:
            # Add the 'label' field
            project['label'] = label

            # If it's not a cheat, add the username
            if username != "cheat_behaviour":
                project['username'] = username
            
            # Insert the modified document into the target collection
            target_collection.insert_one(project)

            
    # Verify and print the data from the target collection
    for username, project_names in user_project_mapping.items():
        print(f"Verifying projects for {username}:")
        query = {"title": {"$in": project_names}}

        # Add a check for the 'username' field if it's not 'cheat_behaviour'
        if username != "cheat_behaviour":
            query['username'] = username

        saved_projects = target_collection.find(query)
        
        for project in saved_projects:
            print(project)
