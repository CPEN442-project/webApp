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

# Global constants for keystrokes
BACKSPACE = 'BACKSPACE'
DELETE = 'DELTE'
ENTER = 'ENTER'
SHIFT_ENTER = 'Shift+ENTER'
CMD_V = 'Cmd+V'
CTRL_V = 'Ctrl+V'
CMD_C = 'Cmd+C'
CTRL_C = 'Ctrl+C'



# Load environment variables
load_dotenv()

app = Flask(__name__)

# Use the MONGO_URI from the .env file
mongo_client = MongoClient(os.getenv('MONGO_URI'))

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


def convert_training_segment_metrics_to_csv(training_segment_metrics, filename="segment_metrics.csv"):
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
        "Ratio_pastes",
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
        "ratio_pastes_overall",
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

                project_copy = project.copy()
                # Remove the '_id' field from the copy
                project_copy.pop('_id', None)


                # Update the document with additional fields
                update = {"$set": {"label": label}}
                project_copy["label"] = label


                if username in ["complete_cheat", "partial_cheat"]:
                    #update["$set"]["cheat_type"] = username
                    project_copy["cheat_type"] = username

                if username == "partial_cheat":
                    # Find the corresponding project in project_names to get cheat periods
                    cheat_project = next((item for item in project_names if item['title'] == project["title"]), None)
                    if cheat_project:
                        #update["$set"]["cheat_periods"] = cheat_project.get("cheat_periods", [])
                        project_copy["cheat_periods"] = cheat_project.get("cheat_periods", [])

                if username not in ["complete_cheat", "partial_cheat"]:
                    #update["$set"]["username"] = username
                    project_copy["username"] = username

                # copy all the fields from the project to the project_copy by using "$set"
                update = {"$set": project_copy}

                # Update or insert the document in the target collection
                target_collection.update_one(query, update, upsert=True)

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
        Ratio_pastes = N_paste_events / N_copy_events if N_copy_events else 0.0
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
            "Ratio_pastes": Ratio_pastes,
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


def create_user_metrics(title, keyboard_events, project=None):
    try:
        # Check if the keyboard_events is a list and not empty
        if not isinstance(keyboard_events, list) or not keyboard_events:
            print(f"Invalid or empty keyboard events for project: {title}")
            return {}, {}  # Return empty dictionaries for both segment and overall metrics

        
        print(f"\nProject title: {title}")
        print(f"Total number of keyboard events: {len(keyboard_events)}")

        events_time_diffs = [time_difference(keyboard_events[i], keyboard_events[i + 1]) for i in range(len(keyboard_events) - 1)]
        segments, segments_cheat_flags = segment_events(title, keyboard_events, events_time_diffs, project)
        #print("segments", segments)

        # Print the number of segments created
        print(f"Number of segments created: {len(segments)}")

        # Variables for time-weighted averages and new metrics
        total_segment_duration = 0.0
        total_final_text_length = 0.0
        total_N_keyboard_events = 0.0
        total_N_keyboard_comb_events = 0.0
        total_N_delete_events = 0.0
        total_N_paste_events = 0.0
        total_N_copy_events = 0.0


        # Variables for time-weighted averages
        weighted_metrics = {
            "error_rate": 0.0,
            "average_consecutive_backspaces": 0.0,
            "cps": 0.0,
            "average_thinking_time": 0.0,
            "pause_frequency": 0.0,
        }
        segment_metrics = []

        for idx, segment in enumerate(segments):
            segment_duration = float(sum(time_difference(segment[i], segment[i + 1]) 
                                        for i in range(len(segment) - 1)))
            total_segment_duration += segment_duration

            segment_time_diffs = [time_difference(segment[i], segment[i + 1]) for i in range(len(segment) - 1)]
            segment_metric = calculate_metrics_for_segment(segment, segment_time_diffs)

            # Add cheat flag to the segment metric
            segment_metric['is_cheat_segment'] = segments_cheat_flags[idx]

            # Add project title and project_id to the segment metric
            segment_metric['title'] = project['title']
            segment_metric['project_id'] = str(project['_id'])

            segment_metrics.append(segment_metric)

            for metric in weighted_metrics:
                weighted_metrics[metric] += segment_metric[metric] * segment_duration

            total_final_text_length += segment_metric["Final_text_length"]
            total_N_keyboard_events += segment_metric["N_keyboard_events"]
            total_N_keyboard_comb_events += segment_metric["N_keyboard_comb_events"]
            total_N_delete_events += segment_metric["N_delete_events"]
            total_N_paste_events += segment_metric["N_paste_events"]
            total_N_copy_events += segment_metric["N_copy_events"]

        overall_metrics = {"title": title}

        overall_metrics.update({
            metric: weighted_metrics[metric] / total_segment_duration if total_segment_duration > 0 else 0.0
            for metric in weighted_metrics
        })

        # Adding new overall metrics
        overall_metrics.update({
            "total_final_text_length": total_final_text_length,
            "average_length_per_event": total_final_text_length / total_N_keyboard_events if total_N_keyboard_events > 0 else 0.0,
            "ratio_combination_overall": total_N_keyboard_comb_events / total_final_text_length if total_final_text_length > 0 else 0.0,
            "ratio_delete_overall": total_N_delete_events / total_final_text_length if total_final_text_length > 0 else 0.0,
            "ratio_pastes_overall": total_N_paste_events / total_N_copy_events if total_N_copy_events > 0 else 0.0,
        })

        return segment_metrics, overall_metrics

    except Exception as e:
        print(f"Error in create_user_metrics for project {title}: ", e)
        traceback.print_exc()  # This prints the traceback of the exception
        return {}, {}



def create_labelled_training_data():
    try:
        # Connect to MongoDB
        connect_to_mongodb(mongo_client)

        # Database and collection setup
        db_name = 'llmdetection'
        collection_name = 'labelled_training_data'
        db = mongo_client[db_name]
        collection = db[collection_name]

        training_data_collection = db['training_data']
        analysis_collection = db['analysis']

        # Retrieve user projects
        user_projects = retrieve_user_projects(db, collection_name)

        training_segment_metrics = []
        all_overall_metrics = []
        for project in user_projects:
            if 'events' not in project:
                continue  # Skip if there are no events

            # Analyze typing behavior
            title = project['title']

            # Analyze typing behavior for a project
            keyboard_events = [event for event in project['events'] if event['type'] in ['keyboard', 'keyboard-combination']]
            mouse_events = [event for event in project['events'] if event['type'] == 'mouse']
            text_state = project.get('textState', '')

            segment_metrics, overall_metrics = create_user_metrics(title, keyboard_events, project)

            # Print metrics
            print_metrics(segment_metrics, overall_metrics)

            # Update MongoDB with segment and overall metrics
            update_query = {'title': title}
            update_data = {
                '$set': {
                    'segment_metrics': segment_metrics,
                    'overall_metrics': overall_metrics
                }
            }
            collection.update_one(update_query, update_data)

            # combine all segment metrics
            training_segment_metrics.append(segment_metrics)
            all_overall_metrics.append(overall_metrics)


        training_data = {
            '$set': {
                'training_segment_metrics': training_segment_metrics,
                'all_overall_metrics': all_overall_metrics,
            }
        }


        training_data_version = "v1.0.1"
        training_query = {'version' : training_data_version}
        training_data_collection.update_one(training_query, training_data, upsert=True)

        convert_training_segment_metrics_to_csv(training_segment_metrics, f"model_and_data/training_segment_metrics_{training_data_version}.csv")
        convert_training_segment_metrics_to_csv(all_overall_metrics, f"model_and_data/all_overall_metrics_{training_data_version}.csv")

    except Exception as e:
        logger.error(f"Error in create_labelled_training_data {title} : {e}")
        traceback.print_exc()  # This prints the traceback of the exception

if __name__ == '__main__':
    # Define the user project names
    user_project_mapping = {
        "so": ["so_actual_work", "so_actual_work2", "so_actual_work3"],
        "april": ["Aj_genuine", "Aj_genuine2"],
        # "aleks": ["Aleks_genuine"], this project has no event
        "roger": ["roger_genuine", "roger_genuine2"],
        "complete_cheat": ["Aleks_fake_1", "Aleks_fake-2", "roger_copypasting", "roger_copypasting2", "Aj_fake_1", "Aj_fake_2", "Shift+Enter test"],
        "partial_cheat": [
            {"title": "so_fake_1", "cheat_periods": [["2023-11-23T04:22:36.761Z", "2023-11-23T04:17:46.109Z"]]},
            {"title": "so_fake_2", "cheat_periods": [["2023-11-23T04:41:21.182Z", "2023-11-23T04:42:36.876Z"]]}
        ]
    }

    transfer_labelled_data(user_project_mapping)

    # This creates the training CSV file
    create_labelled_training_data()

    
    # Initialize the classifier
    classifier = UserBehaviorClassifier(model_version="v1.0.1")

    
    # Final_text_length	N_keyboard_events	N_keyboard_comb_events	Ratio_combination	N_delete_events	Ratio_delete	N_paste_events	N_copy_events	Ratio_pastes	Length_per_event	error_rate	average_consecutive_backspaces	cps	average_thinking_time	pause_frequency	is_cheat_segment
    metrics_columns = ['Final_text_length', 'N_keyboard_events', 'N_keyboard_comb_events', 'Ratio_combination', 'N_delete_events', 'Ratio_delete', 'N_paste_events', 'N_copy_events', 'Ratio_pastes', 'Length_per_event', 'error_rate', 'average_consecutive_backspaces', 'cps', 'average_thinking_time', 'pause_frequency', 'is_cheat_segment']

    # ignore these columns
    # start_time	segment_duration	text_state_change  title	project_id
    ignore_columns = ['start_time', 'segment_duration', 'text_state_change', 'title', 'project_id']

    csv_file_path = "training_segment_metrics.csv"

    from_file = True
    prediction_mode = False

    if not prediction_mode:
        ############# For training mode: #############
        if from_file :
            # Option 1: Load data from CSV
            df = classifier.load_data_from_csv(csv_file_path)
            classifier.train(df, ignore_columns)
            classifier.save_model()
        else:
            # Option 2: Load data from MongoDB
            df = classifier.load_data_from_mongodb('your_collection_name')
            classifier.train(df, ignore_columns)
            classifier.save_model()
    else:
        ############# For prediction mode: #############
        segment_metrics_df = pd.DataFrame(...)  # Replace with your segment metrics DataFrame
        classifier.load_model()
        predictions = classifier.predict(segment_metrics_df)
        print(predictions)



    #app.run(port=3005)


"""
    
    # Initialize the classifier
    # classifier = UserBehaviorClassifier('Synth.csv') # Model already add "model_and_data/" in the path
    classifier = UserBehaviorClassifier('training_segment_metrics.csv') # Model already add "model_and_data/" in the path

    # Train the model
    classifier.train()

    # Make predictions and print them
    predictions = classifier.predict()
    print(predictions[['Actual_Label', 'Predicted_Probability']])

    # Evaluate the model
    classifier.evaluate()

    # Save the model to a file
    classifier.save_model()


    
    # Example usage:
    classifier = UserBehaviorClassifier('Synth.csv')
    classifier.train()
    input_data = pd.DataFrame(...) # Replace with your input data in the same format as the training data
    predictions = classifier.predict(input_data)
    print(predictions[['Predicted_Label', 'Predicted_Probability']])
    classifier.evaluate()
    classifier.save_model()

"""