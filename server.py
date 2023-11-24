from flask import Flask, request
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datetime import datetime
from collections import Counter
from nltk.util import ngrams
import re
import statistics    
from user_behavior_classifier import UserBehaviorClassifier
import numpy as np



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

def transfer_labelled_data(user_project_mapping):
    db_name = 'llmdetection'

    # Source and target collection names
    source_collection_name = 'projects'
    target_collection_name = 'labelled_training_data'

    # Connect to the database and collections
    db = mongo_client[db_name]
    source_collection = db[source_collection_name]
    target_collection = db[target_collection_name]



    # Track newly added items
    newly_added_items = []

    for username, project_names in user_project_mapping.items():
        # Retrieve projects for the current user
        user_projects = source_collection.find({"title": {"$in": project_names}})
        
        # Set the label based on the username
        label = "cheat" if username == "cheat_behaviour" else "genuine"
        
        # Duplicate each project into the target collection with the added label
        for project in user_projects:
            # Prepare the query for checking duplicates
            query = {"title": project["title"]}
            if username != "cheat_behaviour":
                query['username'] = username

            # Check if the document already exists in the target collection
            if target_collection.count_documents(query) == 0:
                # Add the 'label' field
                project['label'] = label

                # If it's not a cheat, add the username
                if username != "cheat_behaviour":
                    project['username'] = username
                
                # Insert the modified document into the target collection
                target_collection.insert_one(project)

                # Add to the newly added items list
                newly_added_items.append(project['title'])

    # Print the titles of the newly added items
    print("Newly added project titles:")
    for title in newly_added_items:
        print(title)

    print("Verification of newly added titles complete.")


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



def find_upper_outliers(time_diffs):
    # Calculate the quartiles
    Q1 = np.percentile(time_diffs, 25)
    Q3 = np.percentile(time_diffs, 75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Determine the outlier upper thresholds
    upper_bound = Q3 + 1.5 * IQR

    # Find upper outliers
    outliers = [x for x in time_diffs if x > upper_bound]
    
    # this will help us segment the session in parts where the user was typing and where he was not
    return outliers


def annotate_segments_with_time_diff(segments, time_diffs):
    annotated_segments = []
    previous_segment_end_time_diff = 0  # For the first segment

    for segment in segments:
        if segment:
            # Annotate the first event of the current segment
            segment[0]['time_diff_to_prev_segment'] = previous_segment_end_time_diff

            # Find the time diff to the next segment (if not the last segment)
            next_segment_index = segments.index(segment) + 1
            next_segment_start_time_diff = time_diffs[len(segment) - 1] if next_segment_index < len(segments) else 0

            # Annotate the last event of the current segment
            segment[-1]['time_diff_to_next_segment'] = next_segment_start_time_diff

            # Update previous_segment_end_time_diff for the next iteration
            previous_segment_end_time_diff = next_segment_start_time_diff

            annotated_segments.append(segment)

    return annotated_segments

def calculate_thinking_time_metrics(segments, time_diffs):
    # this will help us segment the session in parts where the user was typing and where he was not
    outliers = find_upper_outliers(time_diffs)

    average_thinking_times = []
    pause_frequencies = []

    for segment in segments:
        pause_times = []
        for i in range(len(segment) - 1):
            # Check if the time difference is an outlier
            if time_diffs[i] in outliers:
                pause_times.append(time_diffs[i])

        # Calculate average thinking time for this segment
        average_thinking_time = statistics.mean(pause_times) if pause_times else 0
        average_thinking_times.append(average_thinking_time)

        # Calculate frequency of pauses for this segment
        pause_frequency = len(pause_times)
        pause_frequencies.append(pause_frequency)

    return average_thinking_times, pause_frequencies

# Example usage
# segments = segment_events(keyboard_events, time_diffs)
# average_thinking_times, pause_frequencies = calculate_thinking_time_metrics(segments, time_diffs)


def segment_events(keyboard_events, time_diffs):

    # this will help us segment the session in parts where the user was typing and where he was not
    outliers = find_upper_outliers(time_diffs)
    #print("Outliers in time differences:", outliers)

    segments = []
    current_segment = [keyboard_events[0]]

    for i in range(1, len(keyboard_events)):
        if time_diffs[i - 1] in outliers:
            # End the current segment and start a new one
            segments.append(current_segment)
            current_segment = [keyboard_events[i]]
        else:
            # Continue adding to the current segment
            current_segment.append(keyboard_events[i])

    # Add the last segment
    if current_segment:
        segments.append(current_segment)
    
    # Annotate segments with time differences
    annotated_segments = annotate_segments_with_time_diff(segments, time_diffs)

    # the annotated_segments list contains the segments with time in between them, the segments are defined by finding the outliers
    return annotated_segments



def create_user_metrics(keyboard_events, textState):
    # Initialize counters
    backspace_count = 0
    total_keystrokes = 0
    consecutive_backspaces = 0
    backspace_sequences = []

    # Process events
    for event in keyboard_events:
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

    
    # Calculate time differences between consecutive keyboard_events
    time_diffs = [time_difference(keyboard_events[i], keyboard_events[i + 1]) for i in range(len(keyboard_events) - 1)]

    # Segment the keyboard_events
    segments = segment_events(keyboard_events, time_diffs)


    ###### calculate average cps ######
    total_cps = 0
    segment_count = 0

    for segment in segments:
        segment_duration_seconds = sum(time_difference(segment[i], segment[i + 1]) for i in range(len(segment) - 1))
        character_count = sum(1 for event in segment if event['type'] == 'keyboard')
        # Calculate CPS for the segment
        cps = (character_count / segment_duration_seconds) if segment_duration_seconds > 0 else 0
        total_cps += cps
        segment_count += 1

    # Calculate average CPS across all segments ()
    average_cps = total_cps / segment_count if segment_count > 0 else 0

    ###### significant cps changes count, and average cps difference between segments ######
    significant_changes_count, average_cps_difference = detect_significant_cps_change(segments)

    average_thinking_times, pause_frequencies = calculate_thinking_time_metrics(segments, time_diffs)


    return {

        # Top 10 common bigrams the user used, PROBABLY NOT NEEDED
        "most_common_ngrams": (n, most_common_ngrams),

        # number of backspaces over total keystrokes
        "error_rate": error_rate,
        # Average number of consecutive backspaces in the events
        "average_consecutive_backspaces": average_backspace_seq,


        # average character per minute for the entire event array
        "average_cps": average_cps,
        # Number if Calculate significant changes in CPS through out events
        "significant_changes_count": significant_changes_count,
        # Average CPS difference between significant changes
        "average_cps_difference": average_cps_difference,
        # Average thinking time for each segment
        "average_thinking_times": average_thinking_times,
        # Number of pauses for each segment
        "pause_frequencies": pause_frequencies

    }




def create_labelled_training_data(username):
    # Ensure MongoDB connection
    if not check_db_connection(mongo_client):
        raise Exception("Failed to connect to MongoDB.")

    # Database and collection setup
    db_name = 'llmdetection'
    collection_name = 'labelled_training_data'
    if not check_db_and_collection(mongo_client, db_name, collection_name):
        raise Exception(f"Database '{db_name}' or collection '{collection_name}' does not exist.")

    db = mongo_client[db_name]
    collection = db[collection_name]

    # Fetch projects related to the user
    #user_projects = collection.find({"username": username})

    user_projects = collection.find() # get all projects

    for project in user_projects:
        if 'events' not in project:
            continue  # Skip if there are no events

        # Extract keyboard and mouse events
        keyboard_events = [event for event in project['events'] if event['type'] in ['keyboard', 'keyboard-combination']]
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
    # Define the user project names
    # user_project_mapping = {
    #     "so": ["so_actual_work", "so_actual_work2"],
    #     "april": ["Aj_genuine", "Aj_genuine2"],
    #     "aleks": ["Aleks_genuine"],
    #     "roger": ["roger_genuine", "roger_genuine2"],
    #     "cheat_behaviour": ["Aleks_fake_1", "Aleks_fake-2", "roger_copypasting", "roger_copypasting2", "Aj_fake_1", "Aj_fake_2", "so_fake_1", "so_fake_2"]
    # }
    # transfer_labelled_data()


    # Initialize the classifier
    classifier = UserBehaviorClassifier('Synth.csv') # Model already add "model_and_data/" in the path

    # Train the model
    classifier.train()

    # Make predictions and print them
    predictions = classifier.predict()
    print(predictions[['Actual_Label', 'Predicted_Probability']])

    # Evaluate the model
    classifier.evaluate()

    # Save the model to a file
    classifier.save_model()


    

    # username = "so"
    #create_labelled_training_data(username)

    #app.run(port=3005)





