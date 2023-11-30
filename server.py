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
from datetime import timedelta
import matplotlib.pyplot as plt
import difflib
import logging
import traceback  # Add this import at the beginning of your script
from bson import ObjectId  # This is needed to handle ObjectId types
import csv
import os
import pandas as pd
from flask import request, jsonify

# Global constants for keystrokes
BACKSPACE = 'BACKSPACE'
DELETE = 'DELTE'
ENTER = 'ENTER'
SHIFT_ENTER = 'Shift+ENTER'
CMD_V = 'Cmd+V'
CTRL_V = 'Ctrl+V'
CMD_C = 'Cmd+C'
CTRL_C = 'Ctrl+C'

# The server needs to pull new project data from the projects collection
DB_NAME = 'llmdetection'
COLLECTION_NAME = 'projects'


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Use the MONGO_URI from the .env file
mongo_client = MongoClient(os.getenv('MONGO_URI'))
API_PASSWORD = os.getenv('API_PASSWORD')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def connect_to_mongodb(mongo_client):
    # Ensure MongoDB connection
    if not check_db_connection(mongo_client):
        raise Exception("Failed to connect to MongoDB.")

def retrieve_user_projects(db, collection_name):
    # Retrieve user projects from MongoDB
    if collection_name not in db.list_collection_names():
        logger.info(f"Existing collection names: {db.list_collection_names()}.")
        # Collection is created automatically when data is inserted

    collection = db[collection_name]
    return collection.find()

    
def get_project_title_by_id(project_id, db_name=DB_NAME, collection_name=COLLECTION_NAME):
    #print("db_name : ", db_name, ", collection_name : ",  collection_name)
    db = mongo_client[db_name]
    collection = db[collection_name]

    project = collection.find_one({"_id": ObjectId(project_id)})
    return project['title'] if project else None

def get_project_id_by_title(title, db_name=DB_NAME, collection_name=COLLECTION_NAME):
    #print("db_name : ", db_name, ", collection_name : ",  collection_name)
    db = mongo_client[db_name]
    collection = db[collection_name]
    
    project = collection.find_one({"title": title})
    return str(project['_id']) if project else None


def convert_training_segment_metrics_to_csv(training_segment_metrics, filename="model_and_data/training_segment_metrics.csv"):
    if not training_segment_metrics:
        print("No data available to write to CSV.")
        return

    # Open the file in write mode
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = None

        for segment_metrics in training_segment_metrics:
            for segment in segment_metrics:
                if isinstance(segment, dict):
                    # Initialize the CSV DictWriter with fieldnames from the first segment
                    if writer is None:
                        fieldnames = segment.keys()
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        writer.writeheader()
                
                    writer.writerow(segment)


def convert_segment_metrics_to_csv(segment_metrics, filename="model_and_data/segment_metrics.csv"):
    if not segment_metrics:
        print("No data available to write to CSV.")
        return

    # Open the file in write mode
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = None

        for segment in segment_metrics:
            if isinstance(segment, dict):
                # Initialize the CSV DictWriter with fieldnames from the first segment
                if writer is None:
                    fieldnames = segment.keys()
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
            
                writer.writerow(segment)


    print(f"CSV file '{filename}' created successfully.")

# Example usage
# Assuming you have a 'segment_metrics' array from your function
# convert_to_csv(segment_metrics)



def print_metrics(segment_metrics, overall_metrics):
    # Print segment metrics
    metrics_to_print = [
        "start_time",
        "segment_duration",
        "text_state_change",

        ###### These metrics are from Aleks and Aprils ######
        "Final_text_length",
        "N_keyboard_events",
        "N_keyboard_comb_events",
        "Ratio_combination",
        "N_delete_events",
        "Ratio_delete",
        "N_paste_events",
        "N_copy_events",
        "Ratio_V_over_C",
        "Length_per_event",

        # Extra segment Metadata
        "start_time",
        "segment_duration",
        "text_state_change",

        # Number of backspaces over total keystrokes in the segment
        "error_rate",
        # Average number of consecutive backspaces in the segment
        "average_consecutive_backspaces",
        # Average characters per minute for the entire event array
        "cps",
        # Average thinking time for each segment
        "average_thinking_time",  # Ensure this key is included
        # Number of pauses in this segment
        "pause_frequency",
    ]

    overall_metrics_to_print = [
        "error_rate",
        "average_consecutive_backspaces",
        "cps",
        "average_thinking_time",
        "pause_frequency",
        "total_final_text_length",
        "average_length_per_event",
        "ratio_combination_overall",
        "ratio_delete_overall",
        "Ratio_V_over_C_overall",
    ]


    # logger.info("Segment Metrics:")
    # for segment in segment_metrics:
    #     for metric in metrics_to_print:
    #         logger.info(f"Segment {metric}: {segment[metric]}")
    #     logger.info("\n")

    # Print overall metrics
    logger.info("Overall Metrics:")
    for metric in overall_metrics_to_print:
        logger.info(f"Overall metrics -> {metric}: {overall_metrics[metric]}")
    logger.info("\n")



# Function to calculate the time difference between events
def time_difference(event1, event2):
    try:
        time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
        if 'timestamp' in event1 and 'timestamp' in event2:
            return (datetime.strptime(event2['timestamp'], time_format) - 
                    datetime.strptime(event1['timestamp'], time_format)).total_seconds()
        else:
            return 0  # Default value in case of missing timestamp
    except Exception as e:
        print("Error in time_difference: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return 0  # Default value in case of an error


# Function to check database connection
def check_db_connection(client):
    try:
        # This command forces a round trip to the server
        client.admin.command('ping')
        print("MongoDB connection successful.")
        return True
    except Exception as e:
        print("Error connecting to MongoDB: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
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
    

def analyze_text_ngrams(n, textState):
    try:
        if not isinstance(textState, str) or not isinstance(n, int) or n < 1:
            return []

        words = re.findall(r'\w+', textState)
        ngrams_list = list(ngrams(words, n))

        ngram_counts = Counter(ngrams_list)
        most_common_ngrams = ngram_counts.most_common(10)

        return most_common_ngrams
    except Exception as e:
        print("Error in analyze_text_ngrams: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return []
    


def transfer_labelled_data(user_project_mapping):
    try:
        if not isinstance(user_project_mapping, dict):
            return

        db_name = 'llmdetection'

        # Source and target collection names
        source_collection_name = 'projects'
        target_collection_name = 'labelled_training_data'

        # Connect to the database and collections
        db = mongo_client[db_name]
        source_collection = db[source_collection_name]
        target_collection = db[target_collection_name]
        
        for username, project_names in user_project_mapping.items():
            # when partial_cheat, project_names is in the {'title': '...', 'cheat_periods': [...]} format
            if username == "partial_cheat":
                project_titles = [project['title'] for project in project_names]
            else:
                project_titles = project_names

            # Retrieve projects for the current user
            user_projects = source_collection.find({"title": {"$in": project_titles}})
            
            # Set the label based on the username
            label = "cheat" if username in ["complete_cheat", "partial_cheat"] else "genuine"

            for project in user_projects:
                print("project : ", project['title'])

                # Update query to find the document
                query = {"title": project["title"]}

                # Update the document with additional fields
                update = {"$set": {"label": label}}

                if username in ["complete_cheat", "partial_cheat"]:
                    update["$set"]["cheat_type"] = username

                if username == "partial_cheat":
                    # Find the corresponding project in project_names to get cheat periods
                    cheat_project = next((item for item in project_names if item['title'] == project["title"]), None)
                    if cheat_project:
                        update["$set"]["cheat_periods"] = cheat_project.get("cheat_periods", [])

                if username not in ["complete_cheat", "partial_cheat"]:
                    update["$set"]["username"] = username

                # Update or insert the document in the target collection
                target_collection.update_one(query, update, upsert=True)


        print("Verification of newly added titles complete.")
    except Exception as e:
        print("Error in transfer_labelled_data: ", e)
        traceback.print_exc()  # This prints the traceback of the exception


def detect_significant_cps_change(segment_metrics):
    try:
        if not isinstance(segment_metrics, list) or not segment_metrics:
            return 0, 0

        #print("segment_metrics in detect_significant_cps_change()", segment_metrics)

        # Calculate the differences between consecutive cps values
        cps_differences = [abs(segment_metrics[i]['cps'] - segment_metrics[i - 1]['cps']) for i in range(1, len(segment_metrics))]

        # Find outliers in cps differences
        significant_changes = find_upper_outliers(cps_differences)

        # Calculate total and average cps difference for outliers
        total_cps_difference = sum(significant_changes)
        average_cps_difference = total_cps_difference / len(significant_changes) if significant_changes else 0

        return len(significant_changes), average_cps_difference

    except Exception as e:
        print("Error in detect_significant_cps_change: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return 0, 0
    


def find_upper_outliers(time_diffs, multiplier=2):
    try:        
        if not time_diffs:  # Check if the list is empty
            return []  # Return an empty list if there are no time differences

        if not all(isinstance(x, (int, float)) for x in time_diffs):
            return []
        
            
        """
        Identify upper outliers in a dataset based on the interquartile range.
        """
        # Calculate the quartiles
        Q1 = np.percentile(time_diffs, 25)
        Q3 = np.percentile(time_diffs, 75)

        # Calculate the IQR
        IQR = Q3 - Q1

        # Determine the outlier upper thresholds
        upper_bound = Q3 + multiplier * IQR

        # Find upper outliers
        outliers = [x for x in time_diffs if x > upper_bound]

        # this will help us segment the session in parts where the user was typing and where he was not
        return outliers
    
    except Exception as e:
        print("Error in find_upper_outliers: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return []


def calculate_start_time(segment):
    return segment[0]['timestamp'] if segment else None


def calculate_segment_duration(segment_time_diffs):
    return sum(segment_time_diffs)


def calculate_text_state_change(segment):
    if not segment:
        return ''

    initial_text_state = segment[0].get('textState', '')
    final_text_state = segment[-1].get('textState', '')


    # Create a sequence matcher with the initial and final text states
    sm = difflib.SequenceMatcher(None, initial_text_state, final_text_state)


    # Find the first point of divergence
    divergence_point = None
    for opcode in sm.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag != 'equal':
            divergence_point = (i1, j1)
            break

    debug = False
    if divergence_point and debug:
        i, j = divergence_point
        print("------ Debugging Text State Change ------")
        print("Divergence Point in Initial State:", i)
        print("Divergence Point in Final State:", j)
        print("Initial State at Divergence:\n", initial_text_state[i:])
        print("Final State at Divergence:\n", final_text_state[j:])
        print("------ End of Debugging ------\n")

    added_text = ''
    for opcode in sm.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag == 'insert':  # We're only interested in 'insert' operations
            added_text += final_text_state[j1:j2]

    return added_text


def calculate_error_rate(segment):
    backspace_count = sum(1 for event in segment if event['data'] in [BACKSPACE, DELETE])
    character_count = sum(1 for event in segment if event['type'] == 'keyboard')
    return float(backspace_count) / character_count if character_count > 0 else 0.0


def calculate_avg_consecutive_backspaces(segment):
    consecutive_backspaces = 0
    backspace_sequences = []

    for event in segment:
        if event['data'] in [BACKSPACE, DELETE]:
            consecutive_backspaces += 1
        elif consecutive_backspaces > 0:
            backspace_sequences.append(consecutive_backspaces)
            consecutive_backspaces = 0

    if consecutive_backspaces > 0:  # Handle trailing backspaces
        backspace_sequences.append(consecutive_backspaces)

    return float(sum(backspace_sequences)) / len(backspace_sequences) if backspace_sequences else 0.0


def calculate_cps(segment, segment_time_diffs):
    """
    Calculate characters per second (CPS) based on a given segment and time differences between keyboard events.

    Args:
    - segment (list): List of events, each represented as a dictionary. Each event should have a 'type' key.
    - time_diffs (list): List of time differences between consecutive keyboard events.

    Returns:
    - float: Characters per second.
    """
    # Input validation
    if not all(isinstance(event, dict) and 'type' in event for event in segment):
        raise ValueError("Each event in the segment should be a dictionary with a 'type' key.")

    if segment_time_diffs == []:
        return 0.0

    character_count = sum(1 for event in segment if event['type'] == 'keyboard')
    total_time = calculate_segment_duration(segment_time_diffs)

    # Error handling to avoid division by zero
    return float(character_count) / total_time if total_time > 0 else 0.0



def calculate_average_thinking_time(segment_time_diffs, outliers):
    #pause_times = [segment_time_diffs[i] for i in range(len(segment_time_diffs)) if i in outliers]
    pause_length = [time for time in segment_time_diffs if time in outliers]
    return float(np.mean(pause_length) if pause_length else 0.0)


def calculate_pause_frequency(segment_time_diffs, outliers):
    return float(len([time for time in segment_time_diffs if time in outliers]))



def calculate_thinking_time_metrics(segments, time_diffs):
    try:
        if not isinstance(segments, list) or not isinstance(time_diffs, list):
            return []

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
    except Exception as e:
        print("Error in calculate_thinking_time_metrics: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return []
    




# Helper function to parse datetime
def parse_time(t):
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")

# This will also flag the segments based on the project field
# label
# cheat_type
# cheat_periods
def annotate_time_diff_between_segments(segments, time_diffs, project=None):
    try:
        if not isinstance(segments, list) or not isinstance(time_diffs, list):
            return []

        if not segments:
            return  # Handle empty segment list

        cheat_periods = []
        if project and 'cheat_periods' in project:
            # Convert cheat periods to datetime objects for comparison
            cheat_periods = [(parse_time(period[0]), parse_time(period[1])) for period in project['cheat_periods']]

        segments_cheat_flags = []

        # Ensure segments are sorted by the timestamp of their first event
        segments.sort(key=lambda x: parse_time(x[0]['timestamp']))

        for i, segment in enumerate(segments):
            # Parse the start and end time of the segment
            start_time = parse_time(segment[0]['timestamp'])
            end_time = parse_time(segment[-1]['timestamp'])

            # Check if the segment is a cheat segment
            if 'cheat_type' in project:
                if project['cheat_type'] == "complete_cheat":
                    segments_cheat_flags.append(True)
                elif project['cheat_type'] == "partial_cheat":
                    # Check for overlap with cheat periods
                    is_cheat_segment = any(start <= end_time and end >= start_time for start, end in cheat_periods)
                    segments_cheat_flags.append(is_cheat_segment)

            else: # if the cheat_type does not exist, then it is a genuine project
                segments_cheat_flags.append(False)
                

            # Annotate time difference with the previous and next segments
            if i > 0:
                prev_end_time = parse_time(segments[i-1][-1]['timestamp'])
                time_diff_to_prev = (start_time - prev_end_time).total_seconds()
                segment[0]['time_diff_to_prev'] = time_diff_to_prev

            if i < len(segments) - 1:
                next_start_time = parse_time(segments[i+1][0]['timestamp'])
                time_diff_to_next = (next_start_time - end_time).total_seconds()
                segment[-1]['time_diff_to_next'] = time_diff_to_next


        return segments, segments_cheat_flags
    except Exception as e:
        print("Error in annotate_time_diff_between_segments: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return [], []



def segment_events(title, keyboard_events, time_diffs, project=None):
    try:
        if not isinstance(keyboard_events, list) or not isinstance(time_diffs, list):
            return []

        # this will help us segment the session in parts where the user was typing and where he was not
        outliers = find_upper_outliers(time_diffs, 20)
        #print("Outliers in time differences:", outliers)

        plot_histogram = False
        if (plot_histogram):
            plt.figure(figsize=(10, 6))
            plt.hist(time_diffs, bins=30, edgecolor='black', alpha=0.7)
            outliers = find_upper_outliers(time_diffs)
            plt.scatter(outliers, np.zeros_like(outliers), color='red', label='Outliers')
            plt.xlabel('Time Differences')
            plt.ylabel('Frequency')
            plt.title('Distribution of Time Differences with Outliers Highlighted')
            plt.legend()
            plt.show()


        newline_character = [ENTER, SHIFT_ENTER]  # Use global constants
        segments = []
        current_segment = []
        last_event_was_newline = False

        
        for i, event in enumerate(keyboard_events):
            is_newline = event['data'] in newline_character
            is_outlier = i > 0 and time_diffs[i - 1] in outliers
            event['title'] = title
            #event['parent_project_id'] = title

            # Check for segment termination condition
            if is_outlier or (is_newline and not last_event_was_newline):
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                last_event_was_newline = is_newline
            else:
                last_event_was_newline = False

            # Add event to current segment
            current_segment.append(event)

        # Add the last segment
        if current_segment:
            segments.append(current_segment)

        # Annotate segments with time differences
        annotated_segments, segments_cheat_flags = annotate_time_diff_between_segments(segments, time_diffs, project)
        return annotated_segments, segments_cheat_flags

    except Exception as e:
        print("Error in segment_events: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return []


def calculate_metrics_for_segment(segment, segment_time_diffs):
    try:
        if not isinstance(segment, list) or not isinstance(segment_time_diffs, list):
            return None

        start_time = calculate_start_time(segment)
        segment_duration = calculate_segment_duration(segment_time_diffs)
        text_state_change = calculate_text_state_change(segment)

        error_rate = calculate_error_rate(segment)
        avg_consecutive_backspaces = calculate_avg_consecutive_backspaces(segment)
        cps = calculate_cps(segment, segment_time_diffs)

        outlier_multiplier = 0
        outlier = find_upper_outliers(segment_time_diffs, outlier_multiplier)

        plot_histogram_for_segment = False
        title_to_show_plot = "so_fake_1"
        if (plot_histogram_for_segment and title_to_show_plot == segment[0]['title'] and len(segment) > 100):
            plt.figure(figsize=(10, 6))
            plt.hist(segment_time_diffs, bins=30, edgecolor='black', alpha=0.7)
            outliers = find_upper_outliers(segment_time_diffs)
            plt.scatter(outliers, np.zeros_like(outliers), color='red', label='Outliers')
            plt.xlabel('Time Differences')
            plt.ylabel('Frequency')
            plt.title('Distribution of Time Differences with Outliers Highlighted for a single segment')
            plt.legend()
            plt.show()


        avg_thinking_time = calculate_average_thinking_time(segment_time_diffs, outlier)
        pause_frequency = calculate_pause_frequency(segment_time_diffs, outlier)

        

        ###### This metrics is from Aleks and Aprils ######
        Final_text_length = float(len(text_state_change))
        N_keyboard_events = float(sum(1 for event in segment if event['type'] == 'keyboard'))
        N_keyboard_comb_events = float(sum(1 for event in segment if event['type'] == 'keyboard-combination'))
        Ratio_combination = N_keyboard_comb_events / Final_text_length if Final_text_length else 0.0
        N_delete_events = float(sum(1 for event in segment if event['data'] in [BACKSPACE, DELETE]))
        Ratio_delete = N_delete_events / Final_text_length if Final_text_length else 0.0
        N_paste_events = float(sum(1 for event in segment if event['data'] in [CMD_V, CTRL_V]))
        N_copy_events = float(sum(1 for event in segment if event['data'] in [CMD_C, CTRL_C]))
        Ratio_V_over_C = N_paste_events / N_copy_events if N_copy_events else 0.0
        Length_per_event = Final_text_length / (N_keyboard_events + N_keyboard_comb_events) if (N_keyboard_events + N_keyboard_comb_events) else 0.0

        # print("\n\nsegment data")
        # print("###### For Ratio pastes ######")
        # print("N_delete_events", N_delete_events)
        # print("Final_text_length", Final_text_length)
        # print("###### For Ratio pastes ######")
        # print("N_paste_events", N_paste_events)
        # print("N_copy_events", N_copy_events)


        return {
            ###### This metrics is from Aleks and Aprils ######
            "Final_text_length": Final_text_length,
            "N_keyboard_events": N_keyboard_events,
            "N_keyboard_comb_events": N_keyboard_comb_events,
            "Ratio_combination": Ratio_combination,
            "N_delete_events": N_delete_events,
            "Ratio_delete": Ratio_delete,
            "N_paste_events": N_paste_events,
            "N_copy_events": N_copy_events,
            "Ratio_V_over_C": Ratio_V_over_C,
            "Length_per_event": Length_per_event,

            # Extra segment Metadata
            "start_time": start_time,
            "segment_duration": segment_duration,
            "text_state_change": text_state_change,

            # number of backspaces over total keystrokes in the segment
            "error_rate": error_rate,
            # Average number of consecutive backspaces in the segment
            "average_consecutive_backspaces": avg_consecutive_backspaces,
            # average character per minute for the entire event array
            "cps": cps,
            # Average thinking time for each segment
            "average_thinking_time": avg_thinking_time,  # Ensure this key is included
            # Number of pauses in this segment
            "pause_frequency": pause_frequency,

        }

    except ZeroDivisionError as zde:
        print("Division by zero error in calculate_metrics_for_segment: ", zde)
        return None
    except Exception as e:
        print("Error in calculate_metrics_for_segment: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return None


def analysis(project_id, model_version="v1.1.0", model_threshold=0.3, expected_columns=[], ignore_columns=[]):
    try:
        # Connect to MongoDB
        db = mongo_client['llmdetection']
        collection = db['projects']

        # Retrieve project data by project_id
        project = collection.find_one({"_id": ObjectId(project_id)})
        if not project:
            return "Project data not found for the given project_id", 404

        # Process project data to create segments
        keyboard_events = [event for event in project['events'] if event['type'] in ['keyboard', 'keyboard-combination']]
        time_diffs = [time_difference(keyboard_events[i], keyboard_events[i + 1]) for i in range(len(keyboard_events) - 1)]
        segments, _ = segment_events(project['title'], keyboard_events, time_diffs, project)

        # Calculate metrics for each segment
        segment_metrics = [calculate_metrics_for_segment(segment, [time_difference(segment[i], segment[i + 1]) for i in range(len(segment) - 1)]) for segment in segments]

        segment_metrics_ORIGIN = segment_metrics.copy()

        # #print fields in segment_metrics
        # for segment in segment_metrics:
        #     print("\n\nsegment_metric : ", segment,"\n\n")
    
        # Use 1.0.1 because I added actual_work3.csv to the training data
        classifier = UserBehaviorClassifier(model_version=model_version)

        # Convert segment metrics to pandas DataFrame
        segment_metrics_df = pd.DataFrame(segment_metrics)

        # Drop the specified columns
        segment_metrics_df.drop(columns=ignore_columns, errors='ignore', inplace=True)

        # Verify that the number of columns matches N_columns
        # N_columns = 16 - 1  # account for the label 'is_cheat_segment' column
        if len(expected_columns) + len(ignore_columns) != 20:
            raise ValueError(f"Expected {20} columns, but found {len(expected_columns) + len(ignore_columns)}")
        
        # Verify that all expected columns are present and in correct order
        if list(segment_metrics_df.columns) != expected_columns:
            print("\nsegment_metrics_df.columns : ", segment_metrics_df.columns)
            print("\nexpected_columns : ", expected_columns, '\n')
            raise ValueError("Column mismatch. Expected columns: {}, but found: {}".format(expected_columns, list(segment_metrics_df.columns)))

        # Prepare DataFrame for the model
        input_segment_metrics = segment_metrics_df[expected_columns]

        # Check the columns just before prediction
        # print("Columns used for prediction:", input_segment_metrics.columns)

        # Check for null values
        if input_segment_metrics.isnull().any().any():
            raise ValueError("Null values found in input data")


        classifier.load_model()
        predictions = classifier.predict(input_segment_metrics, threshold=model_threshold)
        print(predictions)


        segment_metrics_df_ORIGIN = pd.DataFrame(segment_metrics_ORIGIN)

        
        # Calculate weights based on the number of keyboard events
        total_keyboard_events = segment_metrics_df_ORIGIN['N_keyboard_events'].sum()
        segment_metrics_df_ORIGIN['weight'] = segment_metrics_df_ORIGIN['N_keyboard_events'] / total_keyboard_events

        # Add the new weights to the predictions DataFrame
        predictions['weight'] = segment_metrics_df_ORIGIN['weight']

        # print("predictions['weight'] : ", predictions['weight'])

        # Calculate weighted prediction probabilities
        predictions['weighted_prob'] = predictions['Predicted_Probability'] * predictions['weight']

        # # Calculate weighted cheat detection score
        # # This is depricated because it did not make the score based on how close the prediction probability is to the threshold
        # weighted_cheat_detection_score = (1 - predictions['weighted_prob'].sum()) * 100


        
        # # Classify each segment and calculate a score
        # # Score reflects how much each segment deviates from the threshold
        # predictions['segment_score'] = predictions.apply(
        #     lambda x: (x['Predicted_Probability'] - model_threshold) * x['weight'] if x['Predicted_Probability'] >= model_threshold else (model_threshold - x['Predicted_Probability']) * x['weight'], axis=1)

        # # Calculate the weighted cheat detection score
        # weighted_cheat_detection_score = predictions['segment_score'].sum() * 100

        


        # # Classify each segment and calculate a score
        # # Segments above the threshold are classified as 'Genuine', below as 'Fake'
        # predictions['segment_score'] = predictions.apply(
        #     lambda x: (model_threshold - x['Predicted_Probability']) * x['weight'] * 1 if x['Predicted_Probability'] < model_threshold else (x['Predicted_Probability'] - model_threshold) * x['weight'] * -1, axis=1)

        # # Normalize the weighted cheat detection score to be out of 100%
        # total_possible_score = predictions['weight'].sum() * (1 - model_threshold)
        # weighted_cheat_detection_score = (predictions['segment_score'].sum() / total_possible_score) * 100



        # Add weights to the predictions DataFrame
        predictions['weight'] = segment_metrics_df_ORIGIN['weight']

        # Classify each segment and calculate a score
        predictions['classified'] = predictions['Predicted_Probability'].apply(lambda x: 1 if x >= model_threshold else 0)
        predictions['segment_score'] = predictions['classified'] * predictions['weight']

        # print("predictions['classified'] : ", predictions['classified'])

        # Calculate the weighted cheat detection score
        weighted_cheat_detection_score = predictions['segment_score'].sum() * 100 / predictions['weight'].sum()


        


        # Merge predictions with original segment metrics
        # Ensure the indices align before merging
        segment_metrics_df_ORIGIN = segment_metrics_df_ORIGIN.reset_index(drop=True)
        predictions = predictions.reset_index(drop=True)
        merged_segment_metrics = pd.concat([segment_metrics_df_ORIGIN, predictions[['Predictions', 'Predicted_Probability', 'weight']]], axis=1)

        # Convert merged DataFrame to a list of dictionaries
        merged_segment_metrics_dict = merged_segment_metrics.to_dict(orient='records')



        print("\nsaving the prediction result into local file, or mongoDB\n")

        analysis_collection = "analysis"
        localTest = False
        if localTest:
            # Save the segment metrics to CSV
            convert_segment_metrics_to_csv(segment_metrics_ORIGIN, filename="model_and_data/postman_test_segment_metrics.csv")

            # Save the predictions to CSV
            predictions.to_csv("model_and_data/postman_test_predictions.csv")

            print("project['title'] : ", project['title'])
            print("cheat_detection_score", weighted_cheat_detection_score, "\n")
            # print("merged_segment_metrics_predict", merged_segment_metrics_dict)

            return "Analysis complete. Segment metrics and predictions saved to CSV files."
        else:
            # save the prediction result to mongoDB analysis collection
            analysis_collection = db[analysis_collection]
            # Generate a human-readable timestamp
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Prepare the update data with the new 'last_updated' field
            update_data = { 
                "$set" : {
                    "project_id": project_id,
                    "title": project['title'],
                    "predictions": predictions[['Predictions', 'Predicted_Probability', 'weight']].to_dict(orient='records'),
                    "cheat_detection_score": weighted_cheat_detection_score,
                    "segment_metrics_predict" : merged_segment_metrics_dict,
                    "last_updated": current_timestamp  # Add the timestamp here
                }
            }
            analysis_collection.update_one({"project_id": project_id}, update_data, upsert=True)
            
            print("project['title'] : ", project['title'])
            print("cheat_detection_score", weighted_cheat_detection_score, "\n")
            # print("merged_segment_metrics_predict", merged_segment_metrics_dict)

            return predictions[['Predictions', 'Predicted_Probability', 'weight']].to_json(orient='records')

    except Exception as e:
        print("Error in analysis() : ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return f"An error occurred: {str(e)}", 500

"""

        # Convert segment metrics to pandas DataFrame
        df = pd.DataFrame(segment_metrics)

        # Prepare DataFrame for the model
        input_columns = ['column1', 'column2', ...]  # Replace with actual column names
        input_data = df[input_columns]

        # Load and use the classifier to make predictions
        classifier = UserBehaviorClassifier('training_segment_metrics.csv')
        classifier.load_model()
        prediction_results = classifier.predict(input_data)

        # Return the prediction results
        return prediction_results.to_json(orient='records')
"""
    

@app.route('/analysis', methods=['POST'])  # Change method to POST to include password in request body
def classify_project():
    # Extract password from request body
    req_data = request.get_json()
    password = req_data.get('password') if req_data else None


    # These are the input values for the model
    model_threshold = 0.3
    model_version = "v1.1.0"

    # 15 columns
    # expected__feature_columns = ['Final_text_length', 'N_keyboard_events', 'N_keyboard_comb_events', 'Ratio_combination', 'N_delete_events', 'Ratio_delete', 'N_paste_events', 'N_copy_events', 'Ratio_V_over_C', 'Length_per_event', 'error_rate', 'average_consecutive_backspaces', 'cps', 'average_thinking_time', 'pause_frequency']

    # 5 columns
    # ignore_columns = ['start_time', 'segment_duration', 'text_state_change', 'title', 'project_id']

    # 11 columns
    expected__feature_columns = ['Final_text_length', 'N_keyboard_events', 'Ratio_combination', 'Ratio_delete','N_paste_events' ,'N_copy_events', 'Length_per_event', 'average_consecutive_backspaces', 'cps', 'average_thinking_time', 'pause_frequency']

    # 9 columns
    ignore_columns = ['start_time', 'segment_duration', 'text_state_change', 'N_keyboard_comb_events','N_delete_events', 'Ratio_V_over_C', 'error_rate', 'title', 'project_id']


    # Check if password is correct
    if password != API_PASSWORD:
        return jsonify({"error": "Access denied. Incorrect password."}), 403

    # Continue with the analysis if password is correct
    project_id = req_data.get('project_id')
    if project_id:
        result = analysis(project_id, model_version, model_threshold, expected__feature_columns, ignore_columns)
        return jsonify(result)
    else:
        return jsonify({"error": "Project ID not provided"}), 400
    


# @app.route('/analysis/flag_segments', methods=['POST'])  # Change method to POST to include password in request body
# def classify_project():
#     # Extract password from request body
#     req_data = request.get_json()
#     password = req_data.get('password') if req_data else None

#     # Check if password is correct
#     if password != API_PASSWORD:
#         return jsonify({"error": "Access denied. Incorrect password."}), 403

#     # Continue with the analysis if password is correct
#     project_id = req_data.get('project_id')
#     cheat_periods = req_data.get('cheat_periods')
#     if project_id:
#         pass
#         # result = analysis(project_id)
#         # return jsonify(result)
#     else:
#         return jsonify({"error": "Project ID not provided"}), 400
    
#     start_time = None
#     end_time = None
#     if isinstance(cheat_periods, list):
#         for period in cheat_periods:
#             # if period is a valid pair of datetime strings, use parse_time to convert them to datetime objects
#             if len(period) == 2 and isinstance(period[0], str) and isinstance(period[1], str):
#                 start_time = parse_time(period[0])
#                 end_time = parse_time(period[1])
#                 flag_segments(project_id, start_time, end_time)
#             else:
#                 return jsonify({"error": "Invalid cheat periods format"}), 400
#     else:
#         return jsonify({"error": "cheat_periods has invalid type"}), 400


    
@app.route('/')
def welcome():
    return "Welcome to CPEN442 project API endpoint, please provide necessary credentials to access our endpoints :)"



app.run(port=3005)