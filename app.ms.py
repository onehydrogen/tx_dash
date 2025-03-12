# app.py
import pathlib
import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc,html,dash_table,ctx
from dash.dependencies import Input,Output,State
import pandas as pd
import logging
import sys
import re
import base64
from collections import defaultdict
from datetime import datetime
from typing import Dict,List,Optional,Union
from dotenv import load_dotenv
import tempfile
import traceback
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Get base64 encoded credentials from environment variable
credentials_base64 = os.environ.get('GOOGLE_CREDENTIALS_BASE64')
if credentials_base64:
    # Decode and create a temporary file
    credentials_json = base64.b64decode(credentials_base64).decode('utf-8')
    with open('/tmp/google-credentials.json','w') as f:
        f.write(credentials_json)

    # Set the environment variable to the file path
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/tmp/google-credentials.json'

# Mississippi flag colors
MISSISSIPPI_BLUE = "#001A57"  # Dark blue
MISSISSIPPI_GOLD = "#D2B447"  # Gold/yellow
MISSISSIPPI_RED = "#BF0D3E"  # Red

# Load environment variables
load_dotenv()

# Constants
PAGE_SIZE = int(os.getenv('PAGE_SIZE','10'))
API_KEY = os.getenv('API_KEY','292df5e11916f68de328be25cd942133')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID','1XS261QQNcsuVyW-kcdRlxSRAHp1TfIPA4ZEMojY2cz0')
CREDENTIALS_PATH = os.getenv('CREDENTIALS_PATH','/Users/bendw/Downloads/appleseeddash-ca815365aeef.json')
DEBUG_MODE = os.getenv('DEBUG_MODE','False').lower() == 'true'

# Path configuration
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('legislative_dashboard_debug.log')
    ]
)
logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass


def format_hearing_date(hearing_str):
    """Formats hearing dates for consistent display."""
    try:
        if pd.isna(hearing_str) or hearing_str == 'N/A':
            return 'N/A'

        try:
            date_obj = datetime.strptime(hearing_str,'%Y-%m-%d %H:%M:%S')
            return date_obj.strftime('%b %d, %Y %I:%M %p')
        except:
            return hearing_str
    except Exception as e:
        logger.error(f"Error formatting hearing date: {e}")
        return hearing_str


def standardize_mississippi_status(last_action):
    """
    Standardizes bill status values based on Mississippi-specific terminology
    in the last_action field.
    """
    if pd.isna(last_action):
        return 'active'

    last_action = str(last_action).lower().strip()

    # MS-specific terminology for passed bills
    if any(term in last_action for term in [
        'is now act','became act','approved by the governor','signed by governor',
        'became law without signature'
    ]):
        return 'passed'

    # MS-specific terminology for failed bills
    if any(term in last_action for term in [
        'died in house','died in senate','died in committee',
        'sine die','withdrawn','failed','vetoed','died on calendar',
        'died on motion to reconsider'
    ]):
        return 'failed'

    # For MS, bills can be "held on calendar" which is a special status
    if 'held on calendar' in last_action:
        return 'held'

    # Default to active for anything else
    return 'active'


def get_sample_data():
    """Returns sample data when actual data is unavailable."""
    return pd.DataFrame({
        'year': ['2025','2025'],
        'bill_number': ['HB1234','SB5678'],
        'title': ['Sample Bill 1','Sample Bill 2'],
        'primary_sponsors': ['John Doe','Jane Smith'],
        'party': ['D','R'],
        'district': ['District 1','District 2'],
        'status': ['active','active'],
        'last_action': ['Filed','In Committee'],
        'chamber': ['House','Senate'],
        'latest_hearings': ['N/A','N/A'],
        'past_hearings': ['N/A','N/A'],
        'state_bill_link': ['','']
    })


def load_local_csv():
    """Loads CSV data from local data directory."""
    try:
        # Check for CSV files in the data directory
        csv_files = list(DATA_PATH.glob('*.csv'))

        if not csv_files:
            logger.warning("No CSV files found in data directory, using sample data")
            return get_sample_data()

        # Use the most recent CSV file if multiple exist
        latest_csv = max(csv_files,key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading CSV from: {latest_csv}")

        df = pd.read_csv(latest_csv)
        if df.empty:
            logger.warning("Empty CSV file, using sample data")
            return get_sample_data()

        return parse_contents(df)

    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return get_sample_data()


def setup_credentials_from_env():
    """Sets up Google credentials from environment variables for Heroku deployment."""
    try:
        # Check if GOOGLE_CREDENTIALS environment variable exists
        google_creds = os.getenv('GOOGLE_CREDENTIALS')
        if google_creds:
            # Write the credentials to a temporary file
            with tempfile.NamedTemporaryFile(mode='w+',delete=False,suffix='.json') as temp_file:
                temp_file.write(google_creds)
                temp_path = temp_file.name
                logger.info(f"Created temporary credentials file at {temp_path}")
                return temp_path

        # If no env var but credentials path exists, use it
        if os.path.exists(CREDENTIALS_PATH):
            logger.info(f"Using existing credentials file at {CREDENTIALS_PATH}")
            return CREDENTIALS_PATH

        logger.warning("No Google credentials found")
        return None

    except Exception as e:
        logger.error(f"Error setting up credentials: {e}")
        return None


def load_google_sheets_data():
    """Loads data from Google Sheets using API with enhanced error handling and retries."""
    import time
    import socket
    from ssl import SSLError

    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    logger.info("Attempting to load data from Google Sheets")

    # Get credentials file path
    credentials_path = setup_credentials_from_env()
    if not credentials_path:
        logger.error("No credentials available, using sample data")
        return get_sample_data()

    # Define the scope
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    # Set up credentials
    try:
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path,scope)
        client = gspread.authorize(credentials)
        logger.info("Successfully authenticated with Google")
    except Exception as auth_error:
        logger.error(f"Authentication error: {auth_error}")
        return get_sample_data()

    # Try to access spreadsheet with retries
    for attempt in range(1,MAX_RETRIES + 1):
        try:
            # Open the spreadsheet
            sheet = client.open_by_key(SPREADSHEET_ID).sheet1
            logger.info(f"Successfully opened spreadsheet {SPREADSHEET_ID}")

            # Get all records - with potential for large data
            logger.info(f"Retrieving data (attempt {attempt}/{MAX_RETRIES})...")
            data = sheet.get_all_records()
            logger.info(f"Retrieved {len(data)} records from sheet")

            if not data:
                logger.warning("Empty data from Google Sheets, using sample data")
                return get_sample_data()

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Clean up temporary credentials file if created
            if credentials_path != CREDENTIALS_PATH and os.path.exists(credentials_path):
                os.unlink(credentials_path)
                logger.info("Removed temporary credentials file")

            return parse_contents(df)

        except (SSLError,socket.error,ConnectionError,TimeoutError) as network_error:
            # Handle network-related errors with retry
            logger.warning(f"Network error on attempt {attempt}: {network_error}")
            if attempt < MAX_RETRIES:
                wait_time = RETRY_DELAY * attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {MAX_RETRIES} attempts. Using fallback data.")
                # Try loading from local CSV as fallback
                local_data = load_local_csv()
                if isinstance(local_data,pd.DataFrame) and not local_data.empty:
                    logger.info("Successfully loaded from local CSV fallback")
                    return local_data
                return get_sample_data()

        except Exception as sheet_error:
            logger.error(f"Error accessing spreadsheet: {sheet_error}")
            logger.error(traceback.format_exc())

            # Try loading from local backup if available
            logger.info("Attempting to load from local backup")
            local_data = load_local_csv()
            if isinstance(local_data,pd.DataFrame) and not local_data.empty:
                return local_data
            return get_sample_data()


def parse_contents(df):
    """Parses DataFrame and standardizes the data format."""
    try:
        logger.info(f"Processing DataFrame with columns: {df.columns.tolist()}")

        column_mapping = {
            'bill_number': 'bill_number',
            'title': 'title',
            'last_action': 'last_action',
            'sponsor_name': 'primary_sponsors',
            'sponsor_party': 'party',
            'sponsor_district': 'district',
            'session_year': 'year',
            'state_bill_link': 'state_bill_link',
            'latest_hearings': 'latest_hearings',
            'past_hearings': 'past_hearings'
        }

        df_processed = df.rename(columns=column_mapping)

        # Process hearing dates
        if 'latest_hearings' in df_processed.columns:
            df_processed['latest_hearings'] = df_processed['latest_hearings'].apply(format_hearing_date)
        if 'past_hearings' in df_processed.columns:
            df_processed['past_hearings'] = df_processed['past_hearings'].apply(format_hearing_date)

        # Extract co-sponsors
        if 'co_sponsors' not in df_processed.columns:
            if 'description' in df.columns:
                df_processed['co_sponsors'] = df['description'].apply(
                    lambda x: '; '.join(re.findall(r'Rep/. [A-Za-z/s]+(?:,|$)',str(x))) if pd.notna(x) else 'N/A'
                )
            else:
                df_processed['co_sponsors'] = 'N/A'

        # Determine chamber
        if 'chamber' not in df_processed.columns:
            df_processed['chamber'] = df_processed['bill_number'].apply(
                lambda x: 'House' if str(x).startswith('H') else 'Senate'
            )

        # Standardize status using Mississippi-specific function
        df_processed['status'] = df_processed['last_action'].apply(standardize_mississippi_status)

        # Process links
        if 'state_bill_link' in df_processed.columns:
            df_processed['state_bill_link'] = df_processed['state_bill_link'].apply(
                lambda x: f"[View Bill]({x})" if pd.notna(x) else ""
            )
            df_processed['bill_number'] = df_processed.apply(
                lambda
                    row: f"[{row['bill_number']}]({row['state_bill_link'].replace('[View Bill]','').replace('(','').replace(')','')})"
                if pd.notna(row['state_bill_link']) and row['state_bill_link'] != ""
                else row['bill_number'],
                axis=1
            )

        # Group by bill number
        grouped_df = df_processed.groupby('bill_number').agg({
            'year': 'first',
            'title': 'first',
            'status': 'last',
            'last_action': 'last',
            'primary_sponsors': 'first',
            'co_sponsors': 'first',
            'party': 'first',
            'district': 'first',
            'chamber': 'first',
            'state_bill_link': 'first',
            'latest_hearings': lambda x: '; '.join(filter(lambda v: v != 'N/A',x.unique())),
            'past_hearings': lambda x: '; '.join(filter(lambda v: v != 'N/A',x.unique()))
        }).reset_index()

        return grouped_df.fillna('N/A')

    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return get_sample_data()


def track_mississippi_bill_progress(df):
    """
    Tracks bill progression through Mississippi-specific legislative stages.

    Mississippi legislative process:
    1. Prefiled/Introduced
    2. Referred to Committee
    3. Passed Committee
    4. Floor Action in Original Chamber
    5. Passed Original Chamber
    6. Sent to Second Chamber
    7. Committee Action in Second Chamber
    8. Floor Action in Second Chamber
    9. Passed Second Chamber
    10. Conference Committee (if differences)
    11. Sent to Governor
    12. Governor Action (signed/vetoed)
    """
    try:
        # Initialize counters for Mississippi-specific stages
        progress_stats = {
            'prefiled': 0,
            'in_committee': 0,
            'passed_committee': 0,
            'floor_action': 0,
            'passed_orig_chamber': 0,
            'in_second_chamber': 0,
            'passed_both': 0,
            'sent_to_governor': 0,
            'signed': 0,
            'vetoed': 0,
            'died': 0
        }

        def determine_ms_stage(last_action):
            """Determines Mississippi-specific bill stage based on last action."""
            if pd.isna(last_action):
                return 'prefiled'

            last_action = str(last_action).lower()

            # Governor action
            if any(term in last_action for term in [
                'signed by governor','is now act','became act','approved by the governor'
            ]):
                return 'signed'

            if 'vetoed by governor' in last_action:
                return 'vetoed'

            # Governor consideration phase
            if any(term in last_action for term in [
                'to governor','transmitted to governor','sent to governor'
            ]):
                return 'sent_to_governor'

            # Both chambers passed
            if 'both houses' in last_action or 'enrolled' in last_action:
                return 'passed_both'

            # Second chamber
            if any(term in last_action for term in [
                'transmitted to house','transmitted to senate',
                'received from house','received from senate'
            ]):
                return 'in_second_chamber'

            # Passed original chamber
            if 'passed' in last_action and ('house' in last_action or 'senate' in last_action):
                return 'passed_orig_chamber'

            # Floor action in original chamber
            if any(term in last_action for term in [
                'third reading','second reading','first reading',
                'floor vote','floor debate','on calendar'
            ]):
                return 'floor_action'

            # Committee action
            if 'committee report' in last_action or 'reported out' in last_action:
                return 'passed_committee'

            if any(term in last_action for term in [
                'referred to committee','in committee','committee assignment'
            ]):
                return 'in_committee'

            # Check for died status - MS specific terminology
            if any(term in last_action for term in [
                'died in','died on calendar','sine die','withdrawn','failed'
            ]):
                return 'died'

            # Default to prefiled/introduced
            return 'prefiled'

        for _,row in df.iterrows():
            stage = determine_ms_stage(row['last_action'])
            progress_stats[stage] += 1

        total_bills = len(df)
        progress_percentages = {
            stage: (count / total_bills * 100) if total_bills > 0 else 0
            for stage,count in progress_stats.items()
        }

        return {
            'counts': progress_stats,
            'percentages': progress_percentages
        }

    except Exception as e:
        logger.error(f"Error in track_mississippi_bill_progress: {e}")
        return None


def calculate_mississippi_bill_outcomes(df):
    """Calculates bill outcomes with Mississippi-specific categories."""
    try:
        # Standard outcomes
        status_counts = {
            'passed': 0,
            'failed': 0,
            'active': 0,
            'held': 0  # Mississippi-specific category for bills held on calendar
        }

        # Count bills by Mississippi-specific status
        for _,row in df.iterrows():
            status = standardize_mississippi_status(row['last_action'])
            status_counts[status] += 1

        total_bills = len(df)
        status_percentages = {
            status: (count / total_bills * 100) if total_bills > 0 else 0
            for status,count in status_counts.items()
        }

        return {
            'counts': status_counts,
            'percentages': status_percentages
        }

    except Exception as e:
        logger.error(f"Error calculating Mississippi bill outcomes: {e}")
        return None


def create_mississippi_outcomes_display(outcomes_data):
    """Creates the bill outcomes display component with Mississippi styling."""
    try:
        if not outcomes_data:
            return html.P("No outcome data available",className="text-muted")

        # Mississippi flag color scheme
        ms_blue = "#001A57"  # Dark blue
        ms_gold = "#D2B447"  # Gold/yellow
        ms_red = "#BF0D3E"  # Red

        return html.Div([
            html.Div([
                # Use colors from the Mississippi flag
                html.P([
                    html.Strong("Passed: "),
                    f"{outcomes_data['counts']['passed']} bills ",
                    html.Span(
                        f"({outcomes_data['percentages']['passed']:.1f}%)",
                        className="percentage-badge",
                        style={"background-color": "#2f855a"}  # Green for passed
                    )
                ],className="mb-2"),
                html.P([
                    html.Strong("Failed/Died: "),
                    f"{outcomes_data['counts']['failed']} bills ",
                    html.Span(
                        f"({outcomes_data['percentages']['failed']:.1f}%)",
                        className="percentage-badge",
                        style={"background-color": ms_red}  # MS red for failed
                    )
                ],className="mb-2"),
                html.P([
                    html.Strong("Active: "),
                    f"{outcomes_data['counts']['active']} bills ",
                    html.Span(
                        f"({outcomes_data['percentages']['active']:.1f}%)",
                        className="percentage-badge",
                        style={"background-color": ms_blue}  # MS blue for active
                    )
                ],className="mb-2"),
                html.P([
                    html.Strong("Held on Calendar: "),
                    f"{outcomes_data['counts']['held']} bills ",
                    html.Span(
                        f"({outcomes_data['percentages']['held']:.1f}%)",
                        className="percentage-badge",
                        style={"background-color": ms_gold}  # MS gold for held
                    )
                ],className="mb-2"),
            ],className="pl-4 border-l-4",style={"border-color": ms_gold})
        ])
    except Exception as e:
        logger.error(f"Error creating Mississippi outcomes display: {e}")
        return html.P("Error displaying outcome data",className="text-danger")


def create_mississippi_progress_tracker(progress_data):
    """Creates the progress tracker display component with Mississippi styling."""
    try:
        if not progress_data:
            return html.P("No progress data available",className="text-muted")

        return html.Div([
            html.H5("Bill Progress Breakdown",className="mb-3"),
            html.Div([
                dbc.Card(
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                # Prefiled/Introduced
                                html.P([
                                    html.Strong("Prefiled/Introduced: "),
                                    f"{progress_data['counts']['prefiled']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['prefiled']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),

                                # Committee stages
                                html.P([
                                    html.Strong("In Committee: "),
                                    f"{progress_data['counts']['in_committee']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['in_committee']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),
                                html.P([
                                    html.Strong("Passed Committee: "),
                                    f"{progress_data['counts']['passed_committee']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['passed_committee']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),

                                # Chamber action stages
                                html.P([
                                    html.Strong("Floor Action: "),
                                    f"{progress_data['counts']['floor_action']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['floor_action']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),
                                html.P([
                                    html.Strong("Passed First Chamber: "),
                                    f"{progress_data['counts']['passed_orig_chamber']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['passed_orig_chamber']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),
                                html.P([
                                    html.Strong("In Second Chamber: "),
                                    f"{progress_data['counts']['in_second_chamber']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['in_second_chamber']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),
                                html.P([
                                    html.Strong("Passed Both Chambers: "),
                                    f"{progress_data['counts']['passed_both']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['passed_both']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),

                                # Governor action stages
                                html.P([
                                    html.Strong("Sent to Governor: "),
                                    f"{progress_data['counts']['sent_to_governor']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['sent_to_governor']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),
                                html.P([
                                    html.Strong("Signed into Law: "),
                                    f"{progress_data['counts']['signed']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['signed']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),
                                html.P([
                                    html.Strong("Vetoed: "),
                                    f"{progress_data['counts']['vetoed']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['vetoed']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),

                                # Died bills
                                html.P([
                                    html.Strong("Died: "),
                                    f"{progress_data['counts']['died']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['died']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ],className="mb-2"),
                            ],className="pl-4 border-l-4 border-blue-200")
                        ])
                    ]),
                    className="shadow-sm mb-3"
                )
            ])
        ])
    except Exception as e:
        logger.error(f"Error creating Mississippi progress tracker: {e}")
        return html.P("Error displaying progress data",className="text-danger")


def create_hearings_card(df,search_performed=False):
    """Creates a card displaying latest hearing information."""
    try:
        if not search_performed:
            return html.Div([
                html.H5("Hearing Schedule",className="mb-3"),
                dbc.Card(
                    dbc.CardBody([
                        html.P("Use search bar to display latest hearing info",
                               className="text-muted text-center")
                    ]),
                    className="shadow-sm"
                )
            ])

        if 'latest_hearing_data' not in df.columns:
            return html.P("Latest hearing information not available",className="text-muted")

        latest_hearings = df[df['latest_hearing_data'].notna() & (df['latest_hearing_data'] != 'N/A')]

        return html.Div([
            html.H5("Latest Hearing Information",className="mb-3"),
            dbc.Card(
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.P([
                                html.Strong(row['bill_number']),": ",
                                row['latest_hearing_data']
                            ],className="mb-2")
                            for _,row in latest_hearings.iterrows()
                        ]) if not latest_hearings.empty else html.P("No hearing information available")
                    ])
                ]),
                className="shadow-sm"
            )
        ])
    except Exception as e:
        logger.error(f"Error creating hearings card: {e}")
        return html.P("Error loading hearing information",className="text-danger")


def calculate_sponsor_stats(df,search_value):
    """Calculates comprehensive sponsorship statistics with Mississippi-specific status."""
    try:
        bill_outcomes = defaultdict(int)
        primary_bills = df[df['primary_sponsors'].str.contains(search_value,case=False,na=False)]

        num_primary = len(primary_bills)
        total_bills = len(df)
        primary_percentage = (num_primary / total_bills * 100) if total_bills > 0 else 0

        for _,row in primary_bills.iterrows():
            status = standardize_mississippi_status(row['last_action'])
            bill_outcomes[status] += 1

        # Calculate success rate with Mississippi-specific categories
        completed_bills = bill_outcomes['passed'] + bill_outcomes['failed']
        bill_outcomes['total'] = num_primary
        bill_outcomes['success_rate'] = (
            (bill_outcomes['passed'] / completed_bills * 100)
            if completed_bills > 0 else 0
        )

        progress_analysis = track_mississippi_bill_progress(primary_bills)

        return {
            'primary_bills': num_primary,
            'primary_percentage': primary_percentage,
            'bill_outcomes': bill_outcomes,
            'progress_analysis': progress_analysis
        }

    except Exception as e:
        logger.error(f"Error calculating sponsor stats: {e}")
        return None


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
    ]
)
server = app.server

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Legislative Analytics Dashboard</title>
        {%css%}
        <style>
            /* Mississippi Flag Colors:
               - Dark Blue: #001A57
               - Gold/Yellow: #D2B447 
               - Red: #BF0D3E
            */

            body {
                background-color: #001A57; /* Dark blue background */
            }

            .container-fluid {
                background-color: #FFF; /* White content area for readability */
                padding: 2rem;
                border-radius: 10px;
                margin-top: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .nav-link:hover { 
                background-color: #e9ecef; 
                border-radius: 5px; 
            }

            .card { 
                transition: transform 0.2s;
                border-left: 4px solid #D2B447; /* Gold color from Mississippi flag */
            }

            .card:hover { 
                transform: translateY(-5px); 
            }

            .dashboard-title { 
                background: linear-gradient(120deg, #BF0D3E, #001A57); /* Red to blue gradient */
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                border:4px solid #D2B447; /* Gold border */
            }
            
            .percentage-badge {
                background-color: #BF0D3E; /* Red color */
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 9999px;
                font-size: 0.875rem;
            }
            
            .navbar { 
                background-color: #BF0D3E !important; /* Red color for navbar */
                border-bottom: 3px solid #D2B447; /* Gold border */
            }
            
            .btn-primary { 
                background-color: #D2B447; /* Gold color for primary buttons */
                border-color: #D2B447; 
                color: #001A57; /* Dark blue text */
                font-weight: bold;
            }
            
            .btn-primary:hover { 
                background-color: #B89D3B; /* Darker gold on hover */
                border-color: #B89D3B; 
            }
            
            .btn-secondary {
                background-color: #001A57; /* Dark blue for secondary buttons */
                border-color: #001A57;
            }
            
            .btn-secondary:hover {
                background-color: #00134A; /* Darker blue on hover */
                border-color: #00134A;
            }
            
            .text-primary { 
                color: #001A57 !important; /* Dark blue color for primary text */
            }
            
            /* Card header styling */
            .card-header {
                background-color: #001A57;
                color: white;
            }
            
            /* Table styling */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background-color: #001A57;
                color: white;
                font-weight: bold;
            }
            
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            
            /* Footer styling */
            footer {
                background-color: #BF0D3E;
                color: white;
                padding: 1rem;
                border-top: 3px solid #D2B447;
                margin-top: 2rem;
                border-radius: 0 0 10px 10px;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #D2B447;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #B89D3B;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Navigation bar with logo
navbar = dbc.NavbarSimple(
    children=[

        dbc.NavItem(dbc.NavLink("About", href="https://www.sankofastrategists.com/", target="_blank")),
    ],
    brand=html.Img(src="/assets/ms_flag.png",height="40px"),  # Update this line
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

# Search section
search_section = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Input(
                    id="search-bar",
                    placeholder="Search Bills, Sponsors, or Topics...",
                    type="text",
                    className="mb-3"
                ),
            ], width=8),
            dbc.Col([
                dbc.Button("Search", id="search-button", color="primary", className="mr-2"),
                dbc.Button("Clear", id="clear-search", color="secondary")
            ], width=4)
        ])
    ]),
    className="mb-4"
)


def create_stat_card(title, icon, content):
    return dbc.Card(
        dbc.CardBody([
            html.H4([html.I(className=f"fas {icon} mr-2"), title],
                    className="text-primary"),
            html.Hr(),
            html.Div(id=content, className="mt-3")
        ]),
        className="mb-4 shadow-sm"
    )


# Main layout
app.layout = html.Div([
    dcc.Store(id='original-data'),
    dcc.Location(id='url', refresh=False),

    # Error Alert
    dbc.Alert(
        id='error-message',
        is_open=False,
        duration=5000,
        className="mb-3"
    ),

    navbar,
    dbc.Container([
        # Title section
        html.Div([
            html.H1("Mississippi Legislative Tracker", className="mb-0"),
            html.P("Track, Analyze, and Understand Legislative Data",
                   className="lead mb-0")
        ], className="dashboard-title text-center mb-4"),

        search_section,

        # Stats cards
        dbc.Row([
            dbc.Col(create_stat_card(
                "Sponsorship Overview",
                "fa-users",
                "sponsorship-stats"
            ), md=3),
            dbc.Col(create_stat_card(
                "Bill Outcomes",
                "fa-chart-pie",
                "bill-outcomes"
            ), md=3),
            dbc.Col(create_stat_card(
                "Bill Progress",
                "fa-tasks",
                "progress-tracker"
            ), md=3),
            dbc.Col(create_stat_card(
                "Hearing Schedule",
                "fa-calendar",
                "hearing-schedule"
            ), md=3),
        ]),

        # Bills table
        dbc.Card(
            dbc.CardBody([
                html.H3("Bills Overview", className="mb-4"),
                dash_table.DataTable(
                    id="bills-table",
                    columns=[
                        {"name": "Year", "id": "year"},
                        {"name": "Bill Number", "id": "bill_number", "presentation": "markdown"},
                        {"name": "Title", "id": "title"},
                        {"name": "Primary Sponsor", "id": "primary_sponsors"},
                        {"name": "Party", "id": "party"},
                        {"name": "District", "id": "district"},
                        {"name": "Status", "id": "status"},
                        {"name": "Last Action", "id": "last_action"},
                        {"name": "Chamber", "id": "chamber"},
                        {"name": "Past Hearings", "id": "past_hearings"},
                        {"name": "Link", "id": "state_bill_link", "presentation": "markdown"}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '15px',
                        'fontFamily': '"Segoe UI", sans-serif'
                    },
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold',
                        'border': '1px solid #dee2e6'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        }
                    ],
                    page_size=PAGE_SIZE,
                    page_current=0,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    row_selectable="single",
                    selected_rows=[]
                )
            ]),
            className="mb-4"
        ),

        # Footer
        html.Footer(
            dbc.Row([
                dbc.Col(
                    html.P("© 2024 Sankofa Community Strategists LLC. All rights reserved.",
                           className="text-muted"),
                    className="text-center mt-4"
                )
            ])
        )
    ], fluid=True)
])


# Callbacks
@app.callback(
    [Output("bills-table", "data"),
     Output('sponsorship-stats', 'children'),
     Output('bill-outcomes', 'children'),
     Output('progress-tracker', 'children'),
     Output('hearing-schedule', 'children'),
     Output('original-data', 'data'),
     Output('error-message', 'children'),
     Output('error-message', 'is_open'),
     Output('error-message', 'color')],
    [Input('search-button', 'n_clicks'),
     Input('clear-search', 'n_clicks'),
     Input('url', 'pathname')],
    [State("search-bar", "value"),
     State("original-data", "data"),
     State("bills-table", "data")],
    prevent_initial_call=False
)
def update_dashboard(search_clicks, clear_clicks, pathname, search_value, original_data, current_data):
    """Callback to update dashboard components with Mississippi-specific tracking."""
    triggered_id = ctx.triggered_id if ctx.triggered_id is not None else 'url'

    try:
        # Load initial data if none exists
        if original_data is None:
            # Load data from Google Sheets
            df = load_google_sheets_data()
            if df.empty:
                return [], "No data", "No data", "No data", "No data", None, "No data available", True, "warning"

            # Create initial overview statistics
            overview_stats = html.Div([
                html.P(f"Total Bills: {len(df)}", className="stat-item"),
                html.P(f"Unique Sponsors: {df['primary_sponsors'].nunique()}", className="stat-item")
            ])

            # Mississippi-specific bill outcomes
            ms_outcomes = calculate_mississippi_bill_outcomes(df)
            bill_outcomes = create_mississippi_outcomes_display(ms_outcomes)

            # Mississippi-specific progress tracking
            progress_data = track_mississippi_bill_progress(df)
            progress_tracker = create_mississippi_progress_tracker(progress_data)

            hearing_schedule = create_hearings_card(df)

            return (df.to_dict('records'), overview_stats, bill_outcomes, progress_tracker,
                    hearing_schedule, df.to_dict('records'), None, False, "success")

        # Handle search functionality
        if triggered_id == 'search-button' and search_value:
            df = pd.DataFrame(original_data)
            filtered_df = df[
                df['bill_number'].str.contains(search_value, case=False, na=False) |
                df['title'].str.contains(search_value, case=False, na=False) |
                df['primary_sponsors'].str.contains(search_value, case=False, na=False)
            ]

            if filtered_df.empty:
                return ([], "No results", "No results", "No results", "No results",
                        original_data, f"No results found for '{search_value}'", True, "warning")

            # For sponsor searches, show sponsor-specific stats
            if any(sponsor for sponsor in df['primary_sponsors'].unique()
                   if search_value.lower() in str(sponsor).lower()):

                # Basic sponsorship overview
                sponsor_bills = filtered_df[filtered_df['primary_sponsors'].str.contains(search_value, case=False, na=False)]
                sponsor_name = search_value

                # Count bills with Mississippi-specific status classifications
                ms_sponsor_outcomes = {
                    'passed': 0,
                    'failed': 0,
                    'active': 0,
                    'held': 0
                }

                for _, row in sponsor_bills.iterrows():
                    status = standardize_mississippi_status(row['last_action'])
                    ms_sponsor_outcomes[status] += 1

                total_sponsor_bills = len(sponsor_bills)
                success_rate = (ms_sponsor_outcomes['passed'] / total_sponsor_bills * 100) if total_sponsor_bills > 0 else 0

                stats_display = html.Div([
                    html.P([
                        html.Strong(f"Sponsor: "),
                        sponsor_name
                    ], className="stat-item"),
                    html.P(f"Total Bills: {total_sponsor_bills}", className="stat-item"),
                    html.P(f"Success Rate: {success_rate:.1f}%", className="stat-item")
                ])

                # Sponsor bill outcomes using Mississippi-specific categories
                ms_outcomes = calculate_mississippi_bill_outcomes(sponsor_bills)
                outcomes = create_mississippi_outcomes_display(ms_outcomes)

                # Sponsor bill progress using Mississippi-specific stages
                ms_progress = track_mississippi_bill_progress(sponsor_bills)
                progress = create_mississippi_progress_tracker(ms_progress)

            else:
                # For non-sponsor searches, just show filtered data stats
                overview_stats = html.Div([
                    html.P(f"Results: {len(filtered_df)}", className="stat-item"),
                    html.P(f"Showing bills matching: '{search_value}'", className="stat-item")
                ])

                # Mississippi-specific bill outcomes for filtered results
                ms_outcomes = calculate_mississippi_bill_outcomes(filtered_df)
                outcomes = create_mississippi_outcomes_display(ms_outcomes)

                # Mississippi-specific progress tracking for filtered results
                ms_progress = track_mississippi_bill_progress(filtered_df)
                progress = create_mississippi_progress_tracker(ms_progress)

                stats_display = overview_stats

            hearings = create_hearings_card(filtered_df, search_performed=True)

            return (filtered_df.to_dict('records'), stats_display, outcomes, progress,
                    hearings, original_data, None, False, "success")

        # Handle clear functionality
        elif triggered_id == 'clear-search':
            if original_data:
                df = pd.DataFrame(original_data)
                overview_stats = html.Div([
                    html.P(f"Total Bills: {len(df)}", className="stat-item"),
                    html.P(f"Unique Sponsors: {df['primary_sponsors'].nunique()}", className="stat-item")
                ])

                # Mississippi-specific bill outcomes
                ms_outcomes = calculate_mississippi_bill_outcomes(df)
                bill_outcomes = create_mississippi_outcomes_display(ms_outcomes)

                # Mississippi-specific progress tracking
                progress_data = track_mississippi_bill_progress(df)
                progress_tracker = create_mississippi_progress_tracker(progress_data)

                hearing_schedule = create_hearings_card(df)

                return (df.to_dict('records'), overview_stats, bill_outcomes, progress_tracker,
                        hearing_schedule, original_data, None, False, "success")

        return current_data or [], "No data", "No data", "No data", "No data", original_data, None, False, "success"

    except Exception as e:
        logger.error(f"Error in dashboard update: {e}")
        return (current_data or [], "Error", "Error", "Error", "Error", original_data,
                f"An error occurred: {str(e)}", True, "danger")


if __name__ == '__main__':
    # Use the PORT environment variable for Heroku
    port = int(os.environ.get('PORT', 8051))
    app.run_server(debug=DEBUG_MODE, host='0.0.0.0', port=port)