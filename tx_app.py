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
from flask import Flask, send_from_directory


server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
    ]
)
# Add route to serve static files
@server.route('/assets/<path:path>')
def serve_static(path):
    return send_from_directory('assets', path)

# Texas flag colors
TEXAS_BLUE = "#00205B"  # Dark blue
TEXAS_RED = "#BF0D3E"  # Red
TEXAS_WHITE = "#FFFFFF"  # White

# Load environment variables
load_dotenv()

# Constants
PAGE_SIZE = int(os.getenv('PAGE_SIZE','10'))
API_KEY = os.getenv('API_KEY','292df5e11916f68de328be25cd942133')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID','1kDpbeWjHAgCLsurCWA0QWJaXoKCO2mupq_-UaGSmbUA')
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
        logging.FileHandler('texas_legislative_dashboard_debug.log')
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


def standardize_texas_status(status_desc,last_action):
    """
    Standardizes bill status values based on Texas-specific terminology.
    """
    if pd.isna(status_desc):
        return 'active'

    status_desc = str(status_desc).lower().strip()
    last_action = str(last_action).lower().strip()

    # Texas-specific terminology for passed bills
    if any(term in status_desc for term in ['passed','enrolled','effective','signed']):
        return 'passed'
    if 'sent to governor' in last_action:
        return 'passed'

    # Texas-specific terminology for failed bills
    if any(term in status_desc for term in ['died','failed','vetoed','withdrawn']):
        return 'failed'
    if any(term in last_action for term in ['died','failed','vetoed','withdrawn']):
        return 'failed'

    # Bills in process
    if any(term in status_desc for term in ['engrossed','reported','referred']):
        return 'in_process'

    # Default to active for anything else
    return 'active'


def determine_tx_stage(status_desc,last_action):
    """
    Determines the specific Texas legislative stage based on status and last action.
    """
    if pd.isna(status_desc):
        return 'prefiled'

    status_desc = str(status_desc).lower().strip()
    last_action = str(last_action).lower().strip()

    # Determine stage based on Texas-specific terminology
    if any(term in status_desc for term in ['introduced','filed']):
        return 'prefiled'

    if any(term in status_desc for term in ['referred','committee']):
        return 'in_committee'

    if any(term in status_desc for term in ['reported','favorable']):
        return 'passed_committee'

    if any(term in status_desc for term in ['engrossed','calendar']):
        return 'floor_action'

    if 'passed house' in last_action and ('to senate' in last_action or 'transmitted to senate' in last_action):
        return 'in_second_chamber'
    elif 'passed senate' in last_action and ('to house' in last_action or 'transmitted to house' in last_action):
        return 'in_second_chamber'
    elif any(term in last_action for term in ['passed house','passed senate']):
        return 'passed_orig_chamber'

    if any(term in status_desc for term in ['enrolled','passed both']):
        return 'passed_both'

    if any(term in last_action for term in ['sent to governor','transmitted to governor']):
        return 'sent_to_governor'

    if any(term in status_desc for term in ['effective','signed']):
        return 'signed'

    if 'vetoed' in status_desc or 'vetoed' in last_action:
        return 'vetoed'

    if any(term in status_desc for term in ['died','failed']):
        return 'died'

    return 'in_committee'


def get_sample_data():
    """Returns sample data when actual data is unavailable."""
    return pd.DataFrame({
        'year': ['2023','2023'],
        'bill_number': ['HB1234','SB5678'],
        'title': ['Sample Bill 1','Sample Bill 2'],
        'status_desc': ['Introduced','Engrossed'],
        'last_action': ['Referred to Committee','Reported favorably'],
        'status': ['active','in_process'],
        'stage': ['prefiled','in_committee'],
        'state_link': ['','']
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
    """Parses DataFrame and standardizes the Texas data format."""
    try:
        logger.info(f"Processing DataFrame with columns: {df.columns.tolist()}")

        # Basic column renaming
        column_mapping = {
            'bill_number': 'bill_number',
            'title': 'title',
            'status_desc': 'status_desc',
            'last_action': 'last_action',
            'state_link': 'state_link',
            'status_date': 'status_date'
        }

        df_processed = df.rename(columns=column_mapping)

        # Extract year from bill number or status date
        if 'year' not in df_processed.columns:
            if 'status_date' in df_processed.columns:
                df_processed['year'] = pd.to_datetime(df_processed['status_date']).dt.year.astype(str)
            else:
                df_processed['year'] = '2023'  # Default to current session

        # Standardize status using Texas-specific function
        df_processed['status'] = df_processed.apply(
            lambda row: standardize_texas_status(row['status_desc'],row['last_action']),
            axis=1
        )

        # Determine stage
        df_processed['stage'] = df_processed.apply(
            lambda row: determine_tx_stage(row['status_desc'],row['last_action']),
            axis=1
        )

        # Process links
        if 'state_link' in df_processed.columns:
            df_processed['state_link'] = df_processed['state_link'].apply(
                lambda x: f"[View Bill]({x})" if pd.notna(x) and str(x).startswith('http') else ""
            )
            df_processed['bill_number'] = df_processed.apply(
                lambda
                    row: f"[{row['bill_number']}]({row['state_link'].replace('[View Bill]','').replace('(','').replace(')','')})"
                if pd.notna(row['state_link']) and row['state_link'] != ""
                else row['bill_number'],
                axis=1
            )

        # Fill NA values
        df_processed = df_processed.fillna('N/A')

        # Select and order columns
        final_columns = [
            'year','bill_number','title','status_desc',
            'last_action','status','stage','state_link'
        ]

        # Only include columns that exist in the dataframe
        final_columns = [col for col in final_columns if col in df_processed.columns]

        return df_processed[final_columns]

    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return get_sample_data()


def track_texas_bill_progress(df):
    """
    Tracks bill progression through Texas-specific legislative stages.
    """
    try:
        # Initialize counters for Texas-specific stages
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

        for _,row in df.iterrows():
            stage = row['stage'] if 'stage' in df.columns else determine_tx_stage(row['status_desc'],row['last_action'])
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
        logger.error(f"Error in track_texas_bill_progress: {e}")
        return None


def create_texas_progress_tracker(progress_data):
    """
    Creates the progress tracker display component with Texas styling.
    """
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
        logger.error(f"Error creating Texas progress tracker: {e}")
        return html.P("Error displaying progress data",className="text-danger")


def calculate_texas_bill_outcomes(df):
    """Calculates bill outcomes with Texas-specific categories."""
    try:
        # Standard outcomes
        status_counts = {
            'passed': 0,
            'failed': 0,
            'in_process': 0,
            'active': 0
        }

        # Count bills by Texas-specific status
        for _,row in df.iterrows():
            status = row['status'] if 'status' in df.columns else standardize_texas_status(row['status_desc'],
                                                                                           row['last_action'])
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
        logger.error(f"Error calculating Texas bill outcomes: {e}")
        return None


def create_texas_outcomes_display(outcomes_data):
    """Creates the bill outcomes display component with Texas styling."""
    try:
        if not outcomes_data:
            return html.P("No outcome data available",className="text-muted")

        return html.Div([
            html.Div([
                # Use colors from the Texas flag
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
                        style={"background-color": TEXAS_RED}
                    )
                ],className="mb-2"),
                html.P([
                    html.Strong("In Process: "),
                    f"{outcomes_data['counts']['in_process']} bills ",
                    html.Span(
                        f"({outcomes_data['percentages']['in_process']:.1f}%)",
                        className="percentage-badge",
                        style={"background-color": TEXAS_BLUE}
                    )
                ],className="mb-2"),
                html.P([
                    html.Strong("Active: "),
                    f"{outcomes_data['counts']['active']} bills ",
                    html.Span(
                        f"({outcomes_data['percentages']['active']:.1f}%)",
                        className="percentage-badge",
                        style={"background-color": TEXAS_WHITE,"color": TEXAS_BLUE}
                    )
                ],className="mb-2"),
            ],className="pl-4 border-l-4",style={"border-color": TEXAS_RED})
        ])
    except Exception as e:
        logger.error(f"Error creating Texas outcomes display: {e}")
        return html.P("Error displaying outcome data",className="text-danger")


def calculate_sponsor_stats(df,search_value):
    """Calculates comprehensive sponsorship statistics with Texas-specific status."""
    try:
        bill_outcomes = defaultdict(int)
        primary_bills = df[df['title'].str.contains(search_value,case=False,na=False)]

        num_primary = len(primary_bills)
        total_bills = len(df)
        primary_percentage = (num_primary / total_bills * 100) if total_bills > 0 else 0

        for _,row in primary_bills.iterrows():
            status = row['status'] if 'status' in df.columns else standardize_texas_status(row['status_desc'],
                                                                                           row['last_action'])
            bill_outcomes[status] += 1

        # Calculate success rate with Texas-specific categories
        completed_bills = bill_outcomes['passed'] + bill_outcomes['failed']
        bill_outcomes['total'] = num_primary
        bill_outcomes['success_rate'] = (
            (bill_outcomes['passed'] / completed_bills * 100)
            if completed_bills > 0 else 0
        )

        progress_analysis = track_texas_bill_progress(primary_bills)

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
        <title>Texas Legislative Analytics Dashboard</title>
        {%css%}
        <style>
            /* Texas Flag Colors:
               - Dark Blue: #00205B
               - Red: #BF0D3E 
               - White: #FFFFFF
            */

            body {
                background-color: #00205B; /* Dark blue background */
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
                border-left: 4px solid #BF0D3E; /* Red color from Texas flag */
            }

            .card:hover { 
                transform: translateY(-5px); 
            }

            .dashboard-title { 
                background: linear-gradient(120deg, #00205B, #BF0D3E); /* Blue to red gradient */
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                border: 4px solid white;
            }

            .percentage-badge {
                background-color: #BF0D3E; /* Red color */
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 9999px;
                font-size: 0.875rem;
            }

            .navbar { 
                background-color: #00205B !important; /* Dark blue color for navbar */
                border-bottom: 3px solid #BF0D3E; /* Red border */
            }

            .btn-primary { 
                background-color: #BF0D3E; /* Red color for primary buttons */
                border-color: #BF0D3E; 
                color: white;
                font-weight: bold;
            }

            .btn-primary:hover { 
                background-color: #9A0B32; /* Darker red on hover */
                border-color: #9A0B32; 
            }

            .btn-secondary {
                background-color: #00205B; /* Dark blue for secondary buttons */
                border-color: #00205B;
                color: white;
            }

            .btn-secondary:hover {
                background-color: #001A4A; /* Darker blue on hover */
                border-color: #001A4A;
            }

            .text-primary { 
                color: #00205B !important; /* Dark blue color for primary text */
            }

            /* Card header styling */
            .card-header {
                background-color: #00205B;
                color: white;
            }

            /* Table styling */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background-color: #00205B;
                color: white;
                font-weight: bold;
            }

            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:nth-child(even) {
                background-color: #f2f2f2;
            }

            /* Footer styling */
            footer {
                background-color: #00205B;
                color: white;
                padding: 1rem;
                border-top: 3px solid #BF0D3E;
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
                background: #BF0D3E;
                border-radius: 10px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: #9A0B32;
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
    brand=html.Img(src="/assets/tx_flag.png", height="40px", style={"margin-right": "10px"}),
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
                    placeholder="Search Bills or Topics...",
                    type="text",
                    className="mb-3"
                ),
            ],width=8),
            dbc.Col([
                dbc.Button("Search",id="search-button",color="primary",className="mr-2"),
                dbc.Button("Clear",id="clear-search",color="secondary"),
            ],width=4)
        ])
    ]),
    className="mb-4"
)


def create_stat_card(title,icon,content):
    return dbc.Card(
        dbc.CardBody([
            html.H4([html.I(className=f"fas {icon} mr-2"),title],
                    className="text-primary"),
            html.Hr(),
            html.Div(id=content,className="mt-3")
        ]),
        className="mb-4 shadow-sm"
    )


# Main layout
app.layout = html.Div([
    dcc.Store(id='original-data'),
    dcc.Location(id='url',refresh=False),

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
            html.H1("Texas Legislative Tracker",className="mb-0"),
            html.P("Track, Analyze, and Understand Texas Legislative Data",
                   className="lead mb-0")
        ],className="dashboard-title text-center mb-4"),

        search_section,

        # Stats cards
        dbc.Row([
            dbc.Col(create_stat_card(
                "Legislation Overview",
                "fa-file-alt",
                "legislation-stats"
            ),md=3),
            dbc.Col(create_stat_card(
                "Bill Outcomes",
                "fa-chart-pie",
                "bill-outcomes"
            ),md=3),
            dbc.Col(create_stat_card(
                "Bill Progress",
                "fa-tasks",
                "progress-tracker"
            ),md=3),
            dbc.Col(create_stat_card(
                "Legislative Activity",
                "fa-calendar-alt",
                "activity-stats"
            ),md=3),
        ],className="mb-4"),

        # Bills table
        dbc.Card(
            dbc.CardBody([
                html.H3("Bills Overview",className="mb-4"),
                dash_table.DataTable(
                    id="bills-table",
                    columns=[
                        {"name": "Year","id": "year"},
                        {"name": "Bill Number","id": "bill_number","presentation": "markdown"},
                        {"name": "Title","id": "title"},
                        {"name": "Status","id": "status_desc"},
                        {"name": "Last Action","id": "last_action"},
                        {"name": "Current Status","id": "status"},
                        {"name": "Stage","id": "stage"},
                        {"name": "Link","id": "state_link","presentation": "markdown"}
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
    ],fluid=True)
])


# Callbacks
@app.callback(
    [Output("bills-table","data"),
     Output('legislation-stats','children'),
     Output('bill-outcomes','children'),
     Output('progress-tracker','children'),
     Output('original-data','data'),
     Output('error-message','children'),
     Output('error-message','is_open'),
     Output('error-message','color')],
    [Input('search-button','n_clicks'),
     Input('clear-search','n_clicks'),
     Input('url','pathname')],
    [State("search-bar","value"),
     State("original-data","data"),
     State("bills-table","data")],
    prevent_initial_call=False
)
def update_dashboard(search_clicks,clear_clicks,pathname,search_value,original_data,current_data):
    """Callback to update dashboard components with Texas-specific tracking."""
    triggered_id = ctx.triggered_id if ctx.triggered_id is not None else 'url'

    try:
        # Load initial data if none exists
        if original_data is None:
            # Load data from Google Sheets
            df = load_google_sheets_data()
            if df.empty:
                return [],"No data","No data","No data",{},"No data available",True,"warning"

            # Create initial overview statistics
            overview_stats = html.Div([
                html.P(f"Total Bills: {len(df)}",className="stat-item"),
                html.P(f"Passed Bills: {len(df[df['status'] == 'passed'])}",className="stat-item")
            ])

            # Texas-specific bill outcomes
            tx_outcomes = calculate_texas_bill_outcomes(df)
            bill_outcomes = create_texas_outcomes_display(tx_outcomes)

            # Texas-specific progress tracking
            progress_data = track_texas_bill_progress(df)
            progress_tracker = create_texas_progress_tracker(progress_data)

            return (df.to_dict('records'),overview_stats,bill_outcomes,progress_tracker,
                    df.to_dict('records'),None,False,"success")

        # Handle search and clear operations
        df = pd.DataFrame(original_data)

        if triggered_id == 'clear-search':
            return (df.to_dict('records'),
                    html.Div([
                        html.P(f"Total Bills: {len(df)}",className="stat-item"),
                        html.P(f"Passed Bills: {len(df[df['status'] == 'passed'])}",className="stat-item")
                    ]),
                    create_texas_outcomes_display(calculate_texas_bill_outcomes(df)),
                    create_texas_progress_tracker(track_texas_bill_progress(df)),
                    original_data,
                    "Search cleared",True,"success")

        if triggered_id == 'search-button' and search_value:
            search_value = search_value.strip()
            if not search_value:
                return (df.to_dict('records'),
                        html.Div([
                            html.P(f"Total Bills: {len(df)}",className="stat-item"),
                            html.P(f"Passed Bills: {len(df[df['status'] == 'passed'])}",className="stat-item")
                        ]),
                        create_texas_outcomes_display(calculate_texas_bill_outcomes(df)),
                        create_texas_progress_tracker(track_texas_bill_progress(df)),
                        original_data,
                        "Please enter a search term",True,"warning")

            # Filter the DataFrame based on search value
            filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_value,case=False).any(),axis=1)]

            if filtered_df.empty:
                return (df.to_dict('records'),
                        html.Div([
                            html.P(f"Total Bills: {len(df)}",className="stat-item"),
                            html.P(f"Passed Bills: {len(df[df['status'] == 'passed'])}",className="stat-item")
                        ]),
                        create_texas_outcomes_display(calculate_texas_bill_outcomes(df)),
                        create_texas_progress_tracker(track_texas_bill_progress(df)),
                        original_data,
                        "No results found",True,"warning")

            # Calculate stats for search results
            overview_stats = html.Div([
                html.P(f"Matching Bills: {len(filtered_df)}",className="stat-item"),
                html.P(f"Search Term: {search_value}",className="stat-item")
            ])

            return (filtered_df.to_dict('records'),
                    overview_stats,
                    create_texas_outcomes_display(calculate_texas_bill_outcomes(filtered_df)),
                    create_texas_progress_tracker(track_texas_bill_progress(filtered_df)),
                    original_data,
                    f"Found {len(filtered_df)} results",True,"success")

        # Default return if no specific action
        return (df.to_dict('records'),
                html.Div([
                    html.P(f"Total Bills: {len(df)}",className="stat-item"),
                    html.P(f"Passed Bills: {len(df[df['status'] == 'passed'])}",className="stat-item")
                ]),
                create_texas_outcomes_display(calculate_texas_bill_outcomes(df)),
                create_texas_progress_tracker(track_texas_bill_progress(df)),
                original_data,
                None,False,"success")

    except Exception as e:
        logger.error(f"Error in dashboard update: {e}")
        return (current_data or [],"Error","Error","Error",
                original_data,f"An error occurred: {str(e)}",True,"danger")


if __name__ == '__main__':
    # Use the PORT environment variable for Heroku
    port = int(os.environ.get('PORT',8051))
    app.run_server(debug=DEBUG_MODE,host='0.0.0.0',port=port)