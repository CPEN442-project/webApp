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


def calculate_cpm_for_segments(segments):
    cpm_list = []

    for segment in segments:
        segment_duration_seconds = sum(time_difference(segment[i], segment[i + 1]) for i in range(len(segment) - 1))
        character_count = sum(1 for event in segment if event['type'] == 'keyboard')
        # Calculate CPM for the segment
        cpm = (character_count / segment_duration_seconds) * 60 if segment_duration_seconds > 0 else 0
        cpm_list.append(cpm)

    return cpm_list


def detect_significant_cpm_change(segments, threshold_multiplier=2):
    cpm_list = calculate_cpm_for_segments(segments)
    significant_changes_count = 0
    total_cpm_difference = 0

    for i in range(1, len(cpm_list)):
        cpm_difference = abs(cpm_list[i] - cpm_list[i - 1])
        if cpm_difference > threshold_multiplier * statistics.median(cpm_list):
            significant_changes_count += 1
            total_cpm_difference += cpm_difference

    average_cpm_difference = total_cpm_difference / significant_changes_count if significant_changes_count > 0 else 0

    return significant_changes_count, average_cpm_difference


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

def analyze_typing_behavior(events, textState):
    # Initialize counters
    backspace_count = 0
    total_keystrokes = 0
    consecutive_backspaces = 0
    backspace_sequences = []

    # Regular expression to capture character keys
    char_key_regex = r"Key: ([a-zA-Z])"

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


    ###### calculate average cpm ######
    total_cpm = 0
    segment_count = 0

    for segment in keystroke_segments:
        segment_duration_seconds = sum(time_difference(segment[i], segment[i + 1]) for i in range(len(segment) - 1))
        character_count = sum(1 for event in segment if event['type'] == 'keyboard')
        # Calculate CPM for the segment
        cpm = (character_count / segment_duration_seconds) * 60 if segment_duration_seconds > 0 else 0
        total_cpm += cpm
        segment_count += 1

    # Calculate average CPM across all segments
    average_cpm = total_cpm / segment_count if segment_count > 0 else 0


    significant_changes_count, average_cpm_difference = detect_significant_cpm_change(keystroke_segments)


    return {
        "error_rate": error_rate,
        "average_consecutive_backspaces": average_backspace_seq,
        "most_common_ngrams": (n, most_common_ngrams),
        "average_cpm": average_cpm
    }


def create_user_metrics(username):
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
        project_names = ["april_actual_work", "april_actual_work2"]
    elif username == "aleks":
        project_names = ["aleks", "aleks"]
    elif username == "roger":
        project_names = ["roger", "roger"]

    
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
        user_typing_metrics = analyze_typing_behavior(keyboard_events, textState)

        # Save the results to MongoDB or process further
        # For example, saving the result:
        # result_collection = db['user_metrics']
        # result_collection.insert_one({
        #     "username": username,
        #     "project_title": project['title'],
        #     "typing_metrics": user_typing_metrics
        # })

@app.route('/analysis/project_id', methods=['GET'])
def analyze_project():
    project_id = request.args.get('project_id')
    # http://<hostname>:<port>/analysis?project_id=<value>.
    if project_id:
        result = perform_analysis(project_id)
        return result
    else:
        return "Project ID not provided", 400

if __name__ == '__main__':
    username = "so"
    create_user_metrics(username)
    #app.run(port=3005)
