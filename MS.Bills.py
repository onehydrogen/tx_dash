import requests
import pandas as pd
from datetime import datetime
import os
import zipfile
import json
import time
import shutil
from typing import Dict,List,Optional,Tuple,Set
from tqdm import tqdm
import concurrent.futures
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from google.oauth2 import service_account
from googleapiclient.discovery import build
from bs4 import BeautifulSoup


class BillProcessor:
    def __init__(self):
        self.processed_bills = set()

    def clean_bill_number(self,bill_number: str) -> str:
        """Standardize bill number format"""
        if not isinstance(bill_number,str):
            return str(bill_number)
        return bill_number.strip().upper()

    def process_bills(self,bills_data: List[Dict]) -> List[Dict]:
        """Process bills data to remove duplicates and standardize formatting"""
        processed = []
        seen_bills = set()

        for bill in bills_data:
            bill_number = self.clean_bill_number(bill.get('bill_number',''))
            if bill_number in seen_bills:
                continue
            seen_bills.add(bill_number)
            bill['bill_number'] = bill_number
            processed.append(bill)

        return sorted(processed,key=lambda x: self._bill_sort_key(x['bill_number']))

    def _bill_sort_key(self,bill_number: str) -> tuple:
        """Create sort key for bill numbers"""
        if not bill_number:
            return ('',0)

        parts = bill_number.split()
        if len(parts) != 2:
            return (bill_number,0)

        prefix,number = parts
        try:
            number = int(number)
        except ValueError:
            number = 0

        return (prefix,number)


class LegiscanBulkLoader:
    def __init__(self,api_key: str,output_dir: str = "legiscan_data"):
        self.api_key = api_key
        self.base_url = "https://api.legiscan.com/"
        self.output_dir = output_dir
        os.makedirs(output_dir,exist_ok=True)

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429,500,502,503,504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://",adapter)

    def get_dataset_list(self,state: str = None,year: int = None,special: bool = False) -> Dict:
        """Get available datasets from LegiScan"""
        params = {
            'key': self.api_key,
            'op': 'getDatasetList'
        }

        if state:
            params['state'] = state
        if year:
            params['year'] = year
        if special:
            params['special'] = 1

        response = self.session.get(self.base_url,params=params)
        response.raise_for_status()
        return response.json().get('datasetlist',{})

    def download_dataset(self,dataset_id: str,state: str,year: int) -> str:
        """Download a specific dataset"""
        params = {
            'key': self.api_key,
            'op': 'getDataset',
            'id': dataset_id
        }

        # Create directory for state/year
        dataset_dir = os.path.join(self.output_dir,f"{state}_{year}")
        os.makedirs(dataset_dir,exist_ok=True)

        # Download the dataset
        filename = os.path.join(dataset_dir,f"{state}_{year}_{dataset_id}.zip")
        print(f"Downloading dataset {dataset_id} for {state} {year}...")

        response = self.session.get(self.base_url,params=params,stream=True)
        response.raise_for_status()

        with open(filename,'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return filename

    def extract_dataset(self,zip_filename: str) -> str:
        """Extract the dataset ZIP file"""
        extract_dir = zip_filename.replace('.zip','')

        # Remove the directory if it exists to ensure clean extraction
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)

        os.makedirs(extract_dir,exist_ok=True)

        print(f"Extracting {zip_filename} to {extract_dir}")
        with zipfile.ZipFile(zip_filename,'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        return extract_dir

    def process_dataset(self,dataset_dir: str) -> Tuple[List[Dict],Dict]:
        """Process the extracted dataset"""
        bills = []
        session_info = {}

        # Process session.json first to get session info
        session_file = os.path.join(dataset_dir,'session.json')
        if os.path.exists(session_file):
            with open(session_file,'r') as f:
                session_data = json.load(f)
                session_info = session_data.get('session',{})

        # Process bill JSON files
        bill_files = [f for f in os.listdir(dataset_dir) if f.startswith('bill_') and f.endswith('.json')]
        print(f"Processing {len(bill_files)} bill files from dataset...")

        for filename in tqdm(bill_files):
            bill_path = os.path.join(dataset_dir,filename)
            try:
                with open(bill_path,'r') as f:
                    bill_data = json.load(f)
                    if 'bill' in bill_data:
                        # Add session info to the bill if not present
                        if not bill_data['bill'].get('session') and session_info:
                            bill_data['bill']['session'] = session_info
                        bills.append(bill_data['bill'])
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return bills,session_info


class LegiscanAnalyzer:
    STATE_CODES = {
        'AL': 'Alabama','AK': 'Alaska','AZ': 'Arizona','AR': 'Arkansas',
        'CA': 'California','CO': 'Colorado','CT': 'Connecticut','DE': 'Delaware',
        'FL': 'Florida','GA': 'Georgia','HI': 'Hawaii','ID': 'Idaho',
        'IL': 'Illinois','IN': 'Indiana','IA': 'Iowa','KS': 'Kansas',
        'KY': 'Kentucky','LA': 'Louisiana','ME': 'Maine','MD': 'Maryland',
        'MA': 'Massachusetts','MI': 'Michigan','MN': 'Minnesota','MS': 'Mississippi',
        'MO': 'Missouri','MT': 'Montana','NE': 'Nebraska','NV': 'Nevada',
        'NH': 'New Hampshire','NJ': 'New Jersey','NM': 'New Mexico','NY': 'New York',
        'NC': 'North Carolina','ND': 'North Dakota','OH': 'Ohio','OK': 'Oklahoma',
        'OR': 'Oregon','PA': 'Pennsylvania','RI': 'Rhode Island','SC': 'South Carolina',
        'SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas','UT': 'Utah',
        'VT': 'Vermont','VA': 'Virginia','WA': 'Washington','WV': 'West Virginia',
        'WI': 'Wisconsin','WY': 'Wyoming'
    }

    def __init__(self,api_key: str,output_dir: Optional[str] = None,max_concurrent_requests: int = 10,
                 cache_dir: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.legiscan.com/"
        self.output_dir = output_dir if output_dir else os.getcwd()
        self.cache_dir = cache_dir if cache_dir else os.path.join(self.output_dir,'cache')
        self.max_concurrent_requests = max_concurrent_requests
        self.bill_processor = BillProcessor()

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429,500,502,503,504]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=max_concurrent_requests,
            pool_maxsize=max_concurrent_requests
        )
        self.session.mount("https://",adapter)

        os.makedirs(self.output_dir,exist_ok=True)
        os.makedirs(self.cache_dir,exist_ok=True)

    def _make_request(self,params: dict,use_cache: bool = True) -> dict:
        """Make a request to the Legiscan API with proper error handling and caching"""
        params['key'] = self.api_key

        try:
            response = self.session.get(self.base_url,params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            raise Exception(f"API request failed: {e}")

    def get_available_sessions(self,state: str) -> List[Dict]:
        """Get all available sessions for a state"""
        params = {'op': 'getSessionList','state': state}
        response = self._make_request(params)
        return response.get('sessions',[])

    def get_session_for_year(self,state: str,year: int) -> Optional[Dict]:
        """Get session information for a specific year with session selection"""
        sessions = self.get_available_sessions(state)

        if not sessions:
            print(f"No sessions found for {self.STATE_CODES[state]}.")
            return None

        # Filter sessions for the requested year
        year_sessions = [s for s in sessions if s.get('year_start') == year]

        if not year_sessions:
            print(f"\nNo sessions found for year {year}.")
            return None

        # If multiple sessions exist for the year, let user choose
        if len(year_sessions) > 1:
            print(f"\nMultiple sessions found for {year}. Please select one:")
            for i,session in enumerate(year_sessions,1):
                print(f"{i}. {session.get('name','Unnamed Session')}")

            while True:
                try:
                    choice = int(input("\nEnter the number of your choice: "))
                    if 1 <= choice <= len(year_sessions):
                        selected_session = year_sessions[choice - 1]
                        print(f"\nSelected: {selected_session.get('name')}")
                        return selected_session
                    print("Please enter a valid number.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            # Only one session found
            return year_sessions[0]

    def _check_pdf_link(self,state_link: str) -> Optional[str]:
        """Check the state bill page for a PDF link"""
        try:
            response = self.session.get(state_link,timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text,'html.parser')
            pdf_link = None
            pdf_patterns = [
                'a[href$=".pdf"]',
                'a[href*="PDF"]',
                'a[href*="billtext"]',
                'a[href*="BillText"]'
            ]

            for pattern in pdf_patterns:
                links = soup.select(pattern)
                if links:
                    for link in links:
                        href = link.get('href','').lower()
                        if 'bill' in href and ('pdf' in href or 'text' in href):
                            pdf_link = link['href']
                            break
                    if pdf_link:
                        break
                    pdf_link = links[0]['href']
                    break

            if pdf_link and not pdf_link.startswith('http'):
                pdf_link = requests.compat.urljoin(state_link,pdf_link)
            return pdf_link

        except Exception as e:
            print(f"Error checking PDF link for {state_link}: {e}")
            return None

    def _get_bill_details(self,bill_id: int) -> dict:
        """Get details for a specific bill"""
        response = self._make_request({'op': 'getBill','id': bill_id})
        return response

    def get_hearing_info(self,bill: dict) -> Tuple[str,str]:
        """Extract upcoming and past hearing information from bill history"""
        upcoming_hearings = []
        past_hearings = []
        current_time = datetime.now()

        try:
            # Check if we have a valid history list
            history = bill.get('history',[])
            if not isinstance(history,list):
                print(f"Invalid history data type: {type(history)}")
                return 'N/A','N/A'

            # Keywords that indicate hearing events
            hearing_keywords = [
                'hearing','committee','referred to',
                'reported from committee','scheduled',
                'meeting','testimony'
            ]

            for event in history:
                if not isinstance(event,dict):
                    continue

                action = event.get('action','').lower()
                date_str = event.get('date','')

                # Skip if no date or action
                if not date_str or not action:
                    continue

                # Check if this is a hearing-related event
                is_hearing_event = any(keyword in action for keyword in hearing_keywords)
                if is_hearing_event:
                    try:
                        # First try full datetime format
                        try:
                            hearing_date = datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            # If that fails, try date-only format and set time to midnight
                            try:
                                hearing_date = datetime.strptime(date_str,'%Y-%m-%d')
                            except ValueError as e:
                                print(f"Error parsing date {date_str}: {e}")
                                continue

                        committee_info = ''

                        # Try to extract committee information
                        if 'committee' in action:
                            committee_parts = action.split('committee',1)
                            if len(committee_parts) > 1:
                                committee_info = f" - {committee_parts[0].strip()} Committee"

                        formatted_date = (f"{hearing_date.strftime('%Y-%m-%d %H:%M:%S')}"
                                          f"{committee_info} - {event.get('action','')}")

                        if hearing_date > current_time:
                            upcoming_hearings.append(formatted_date)
                        else:
                            past_hearings.append(formatted_date)
                    except ValueError as e:
                        print(f"Error parsing date {date_str}: {e}")
                        continue

        except Exception as e:
            print(f"Error processing hearing information: {e}")
            return 'N/A','N/A'

        return (
            '; '.join(sorted(upcoming_hearings)) if upcoming_hearings else 'N/A',
            '; '.join(sorted(past_hearings)) if past_hearings else 'N/A'
        )

    def process_bill_data(self,bill: dict) -> Optional[dict]:
        """Process individual bill data for Google Sheets"""
        if not bill or not isinstance(bill,dict):
            return None

        try:
            bill_number = bill.get('bill_number','')
            print(f"\nProcessing bill data for {bill_number}")

            # Get and log history data
            history = bill.get('history',[])
            print(f"Found {len(history)} history events")

            upcoming_hearings,past_hearings = self.get_hearing_info(bill)
            print(f"Upcoming hearings: {upcoming_hearings}")
            print(f"Past hearings: {past_hearings}")

            # Get status information
            status = bill.get('status_desc','')
            if not status:
                # Try to get status from the last history entry
                if history and isinstance(history,list) and history[-1]:
                    status = history[-1].get('action','N/A')

            # Get last action with date
            last_action = 'N/A'
            if history and isinstance(history,list) and history[-1]:
                last_action = history[-1].get('action','N/A')
                last_action_date = history[-1].get('date','')
                if last_action_date:
                    last_action = f"{last_action_date} - {last_action}"

            sponsors = bill.get('sponsors',[])
            primary_sponsor_info = {'name': 'N/A','party': 'N/A','district': 'N/A'}

            if isinstance(sponsors,list) and sponsors:
                for sponsor in sponsors:
                    if isinstance(sponsor,dict) and sponsor.get('sponsor_type_id') == 1:
                        primary_sponsor_info = {
                            'name': sponsor.get('name','N/A'),
                            'party': sponsor.get('party','N/A'),
                            'district': sponsor.get('district','N/A')
                        }
                        break

            state_link = bill.get('state_link','')
            pdf_link = self._check_pdf_link(state_link) if state_link else None
            final_link = pdf_link if pdf_link else state_link

            return {
                'year': str(bill.get('session',{}).get('year_start','')),
                'bill_number': bill_number,
                'title': bill.get('title',''),
                'primary_sponsors': primary_sponsor_info['name'],
                'party': primary_sponsor_info['party'],
                'district': primary_sponsor_info['district'],
                'status': status or 'N/A',
                'last_action': last_action,
                'chamber': 'House' if bill_number.upper().startswith('H') else 'Senate',
                'upcoming_hearings': upcoming_hearings,
                'past_hearings': past_hearings,
                'state_bill_link': final_link
            }

        except Exception as e:
            print(f"Error processing bill {bill.get('bill_number','UNKNOWN')}: {e}")
            return None

    def _process_bills_parallel(self,bills_list: List[dict]) -> List[dict]:
        """Process bills in parallel with deduplication"""
        if not bills_list:
            return []

        processed_rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            future_to_bill = {
                executor.submit(self._get_bill_details,bill['bill_id']): bill
                for bill in bills_list
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_bill),total=len(bills_list)):
                try:
                    response = future.result()
                    if response and 'bill' in response:
                        print(f"\nProcessing response for bill ID: {response['bill'].get('bill_id')}")
                        processed_row = self.process_bill_data(response['bill'])
                        if processed_row:
                            processed_rows.append(processed_row)
                except Exception as e:
                    print(f"Error processing bill: {e}")

        return self.bill_processor.process_bills(processed_rows)

    def get_changed_bills(self,master_list: dict,existing_bill_numbers: Set[str]) -> List[dict]:
        """Identify bills that have changed or are new from the master list"""
        changed_bills = []

        for bill_data in master_list.values():
            if isinstance(bill_data,dict) and 'bill_id' in bill_data:
                bill_number = self.bill_processor.clean_bill_number(bill_data.get('number',''))

                # Add if it's a new bill or has changes
                if bill_number not in existing_bill_numbers:
                    print(f"New bill found: {bill_number}")
                    changed_bills.append(bill_data)

        return changed_bills


class GoogleSheetUpdater:
    def __init__(self,spreadsheet_id: str,credentials_path: str):
        self.spreadsheet_id = spreadsheet_id
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        self.service = build('sheets','v4',credentials=self.credentials)
        self.bill_processor = BillProcessor()

    def get_existing_data(self) -> Tuple[List[List],Dict[str,int],Set[str]]:
        """Get existing data from Google Sheet and create a bill number to row index mapping"""
        sheets = self.service.spreadsheets()

        # Get sheet information to find the right sheet
        sheet_metadata = sheets.get(spreadsheetId=self.spreadsheet_id).execute()
        sheets_info = sheet_metadata.get('sheets',[])

        if not sheets_info:
            return [],{},set()

        # Use the first sheet by default
        sheet_id = sheets_info[0]['properties']['sheetId']
        sheet_name = sheets_info[0]['properties']['title']

        # Get the headers first to determine which column has bill_number
        headers_result = sheets.values().get(
            spreadsheetId=self.spreadsheet_id,
            range=f"{sheet_name}!A1:L1"
        ).execute()
        headers = headers_result.get('values',[[]])

        if not headers:
            # Sheet might be empty, initialize with headers
            headers = [["year","bill_number","title","primary_sponsors","party",
                        "district","status","last_action","chamber",
                        "upcoming_hearings","past_hearings","state_bill_link"]]

            sheets.values().update(
                spreadsheetId=self.spreadsheet_id,
                range=f"{sheet_name}!A1:L1",
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()

            return headers,{},set()

        headers = headers[0]

        # Find bill_number column index
        bill_number_col = -1
        for i,header in enumerate(headers):
            if header.lower() == 'bill_number' or header.lower() == 'bill number':
                bill_number_col = i
                break

        if bill_number_col == -1:
            print("Warning: Could not find bill_number column in the headers!")
            return [headers],{},set()

        # Get all data
        result = sheets.values().get(
            spreadsheetId=self.spreadsheet_id,
            range=f"{sheet_name}!A:L"  # Adjust range based on your actual columns
        ).execute()

        values = result.get('values',[])
        if not values or len(values) <= 1:  # Only headers or empty
            return values,{},set()

        # Create a mapping of bill numbers to row indices and a set of existing bill numbers
        bill_to_row = {}
        existing_bill_numbers = set()

        for i,row in enumerate(values[1:],2):  # Start from index 2 (row 2 in sheets)
            if len(row) > bill_number_col:
                bill_number = self.bill_processor.clean_bill_number(row[bill_number_col])
                bill_to_row[bill_number] = i
                existing_bill_numbers.add(bill_number)

        return values,bill_to_row,existing_bill_numbers

    def format_values(self,processed_bills: List[Dict]) -> List[List]:
        """Format bill data for Google Sheets"""
        formatted_rows = []
        for bill in processed_bills:
            formatted_row = [
                bill.get('year',''),
                bill.get('bill_number',''),
                bill.get('title',''),
                bill.get('primary_sponsors',''),
                bill.get('party',''),
                bill.get('district',''),
                bill.get('status',''),
                bill.get('last_action',''),
                bill.get('chamber',''),
                bill.get('upcoming_hearings',''),
                bill.get('past_hearings',''),
                bill.get('state_bill_link','')
            ]
            formatted_rows.append(formatted_row)
        return formatted_rows

    def update_sheet(self,new_bills: List[Dict],existing_data: List[List],bill_to_row: Dict[str,int]) -> None:
        """Update Google Sheet with new and updated bill information"""
        if not new_bills:
            print("No bills to update")
            return

        sheets = self.service.spreadsheets()

        # For new bills, add to the end of the sheet
        new_bill_rows = []
        update_operations = []

        for bill in new_bills:
            bill_number = self.bill_processor.clean_bill_number(bill['bill_number'])

            # Format the bill data
            formatted_row = [
                bill.get('year',''),
                bill.get('bill_number',''),
                bill.get('title',''),
                bill.get('primary_sponsors',''),
                bill.get('party',''),
                bill.get('district',''),
                bill.get('status',''),
                bill.get('last_action',''),
                bill.get('chamber',''),
                bill.get('upcoming_hearings',''),
                bill.get('state_bill_link',''),
                bill.get('pas_hearings','')
            ]

            if bill_number in bill_to_row:
                # Update existing bill
                row_index = bill_to_row[bill_number]
                update_range = f'A{row_index}:L{row_index}'

                update_operations.append({
                    'range': update_range,
                    'values': [formatted_row]
                })
                print(f"Updating existing bill {bill_number} at row {row_index}")
            else:
                # Add as a new bill
                new_bill_rows.append(formatted_row)
                print(f"Adding new bill {bill_number}")

        # Perform all updates first
        if update_operations:
            batch_body = {
                'valueInputOption': 'RAW',
                'data': update_operations
            }
            sheets.values().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body=batch_body
            ).execute()
            print(f"Updated {len(update_operations)} existing bills")

        # Then append all new bills
        if new_bill_rows:
            append_body = {
                'values': new_bill_rows
            }
            sheets.values().append(
                spreadsheetId=self.spreadsheet_id,
                range='A1',  # It will append after the last row
                valueInputOption='RAW',
                body=append_body
            ).execute()
            print(f"Added {len(new_bill_rows)} new bills")

        print(f"Sheet update complete. Total bills processed: {len(new_bills)}")


def get_user_input() -> Tuple[str,int,bool,int]:
    """Get state, year, continuous monitoring preference and update method from user"""
    print("\nAvailable states:")
    for code,name in LegiscanAnalyzer.STATE_CODES.items():
        print(f"{code}: {name}")

    while True:
        state = input("\nEnter state code (e.g., AR for Arkansas): ").strip().upper()
        if state in LegiscanAnalyzer.STATE_CODES:
            break
        print("Invalid state code. Please try again.")

    while True:
        try:
            year = int(input("\nEnter year (e.g., 2024): ").strip())
            if 2000 <= year <= datetime.now().year + 1:
                break
            print("Please enter a valid year between 2000 and next year.")
        except ValueError:
            print("Please enter a valid year.")

    while True:
        monitor = input("\nContinuously monitor for updates? (yes/no): ").strip().lower()
        if monitor in ['yes','no','y','n']:
            break
        print("Please enter 'yes' or 'no'.")

    print("\nUpdate method options:")
    print("1. Bulk import (recommended for initial load or full refresh)")
    print("2. API calls (slower but can target specific bills)")
    print("3. Hybrid (attempt bulk import first, fall back to API calls if needed)")

    while True:
        try:
            method = int(input("\nEnter your update method choice (1-3): ").strip())
            if 1 <= method <= 3:
                break
            print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number.")

    return state,year,monitor.startswith('y'),method


def process_state_bills_api(legiscan: LegiscanAnalyzer,sheet_updater: GoogleSheetUpdater,
                            state: str,year: int,existing_bill_numbers: Set[str]) -> bool:
    """Process bills for a specific state and year using API calls"""
    print(f"\nProcessing bills for {LegiscanAnalyzer.STATE_CODES[state]} ({year}) using API calls")

    # Get session for the specified year
    session = legiscan.get_session_for_year(state,year)
    if not session:
        return False

    print(f"\nUsing session: {session.get('name','Unknown Session')}")
    print(f"Session period: {session.get('year_start')}-{session.get('year_end',session.get('year_start'))}")

    # Get master list
    params = {'op': 'getMasterList','id': session['session_id']}
    try:
        response = legiscan._make_request(params)

        if not response.get('masterlist'):
            print("No bills found in this session.")
            return False

        # Get existing data from the Google Sheet
        existing_data,bill_to_row,_ = sheet_updater.get_existing_data()

        # Identify bills that have changed
        changed_bills = legiscan.get_changed_bills(response.get('masterlist',{}),existing_bill_numbers)

        if not changed_bills:
            print("No new or changed bills found")
            return True

        # Process the changed bills
        print(f"\nProcessing {len(changed_bills)} bills that are new or changed...")
        processed_rows = legiscan._process_bills_parallel(changed_bills)

        if processed_rows:
            sheet_updater.update_sheet(processed_rows,existing_data,bill_to_row)
            return True

        return False

    except Exception as e:
        print(f"Error fetching bills via API: {e}")
        return False


def process_state_bills_bulk(bulk_loader: LegiscanBulkLoader,legiscan: LegiscanAnalyzer,
                             sheet_updater: GoogleSheetUpdater,state: str,year: int,
                             existing_data: List[List],bill_to_row: Dict[str,int]) -> bool:
    """Process bills for a specific state and year using bulk import"""
    print(f"\nProcessing bills for {LegiscanAnalyzer.STATE_CODES[state]} ({year}) using bulk import")

    # Get available datasets
    datasets = bulk_loader.get_dataset_list(state,year)

    if not datasets:
        print(f"No bulk datasets available for {state} {year}")
        return False

    print(f"Found {len(datasets)} available dataset(s)")

    # Find the most recent dataset
    latest_dataset = None
    latest_timestamp = 0

    # Check if datasets is a dictionary or list
    if isinstance(datasets,dict):
        # If it's a dictionary, iterate through items
        for dataset_id,dataset in datasets.items():
            timestamp = dataset.get('dataset_hash_ts',0)
            if timestamp > latest_timestamp:
                latest_dataset = dataset_id
                latest_timestamp = timestamp
    elif isinstance(datasets,list):
        # If it's a list, iterate through elements
        for dataset in datasets:
            dataset_id = dataset.get('dataset_id')
            timestamp = dataset.get('dataset_hash_ts',0)
            if timestamp > latest_timestamp and dataset_id:
                latest_dataset = dataset_id
                latest_timestamp = timestamp

    if not latest_dataset:
        print("No valid dataset found")
        return False

def process_state_bills_hybrid(bulk_loader: LegiscanBulkLoader,legiscan: LegiscanAnalyzer,
                               sheet_updater: GoogleSheetUpdater,state: str,year: int) -> bool:
    """Process bills using a hybrid approach - try bulk first, fall back to API if needed"""
    print(f"\nProcessing bills for {LegiscanAnalyzer.STATE_CODES[state]} ({year}) using hybrid approach")

    # Get existing data
    existing_data,bill_to_row,existing_bill_numbers = sheet_updater.get_existing_data()

    # Try bulk import first
    bulk_success = process_state_bills_bulk(bulk_loader,legiscan,sheet_updater,state,year,existing_data,bill_to_row)

    if bulk_success:
        print("Bulk import successful")
        # Now check for any updates using the API for completeness
        print("Checking for any additional updates via API...")
        # Get fresh data after bulk import
        _,_,updated_bill_numbers = sheet_updater.get_existing_data()
        api_success = process_state_bills_api(legiscan,sheet_updater,state,year,updated_bill_numbers)
        return True
    else:
        print("Bulk import failed or unavailable. Falling back to API...")
        return process_state_bills_api(legiscan,sheet_updater,state,year,existing_bill_numbers)


def main():
    # Configuration
    print("\n=== LegiScan Bill Tracker Configuration ===")
    API_KEY = input("\nEnter your Legiscan API key: ").strip()
    SPREADSHEET_ID = input("Enter your Google Spreadsheet ID: ").strip()
    CREDENTIALS_PATH = input("Enter path to your Google service account credentials JSON: ").strip()
    UPDATE_INTERVAL = 1800  # 30 minutes in seconds

    # Initialize components
    try:
        legiscan = LegiscanAnalyzer(
            API_KEY,
            max_concurrent_requests=10,
            cache_dir='legiscan_cache'
        )

        bulk_loader = LegiscanBulkLoader(API_KEY)
        sheet_updater = GoogleSheetUpdater(SPREADSHEET_ID,CREDENTIALS_PATH)
        print("\nSuccessfully initialized LegiScan, Bulk Loader, and Google Sheets connection.")
    except Exception as e:
        print(f"\nError during initialization: {e}")
        return

    while True:
        try:
            # Get user input for state and year
            state,year,continuous_monitoring,update_method = get_user_input()

            # Process bills
            while True:
                try:
                    start_time = datetime.now()
                    print(f"\nStarting update at {start_time}")

                    # Use the selected update method
                    if update_method == 1:  # Bulk import
                        existing_data,bill_to_row,_ = sheet_updater.get_existing_data()
                        process_state_bills_bulk(bulk_loader,legiscan,sheet_updater,state,year,existing_data,
                                                 bill_to_row)
                    elif update_method == 2:  # API calls
                        _,_,existing_bill_numbers = sheet_updater.get_existing_data()
                        process_state_bills_api(legiscan,sheet_updater,state,year,existing_bill_numbers)
                    else:  # Hybrid approach
                        process_state_bills_hybrid(bulk_loader,legiscan,sheet_updater,state,year)

                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    print(f"\nUpdate completed in {duration:.1f} seconds")

                    if not continuous_monitoring:
                        break

                    print(f"\nWaiting {UPDATE_INTERVAL / 60} minutes before next update...")
                    time.sleep(UPDATE_INTERVAL)

                except KeyboardInterrupt:
                    print("\nMonitoring stopped by user")
                    break
                except Exception as e:
                    print(f"\nError processing bills: {e}")
                    if not continuous_monitoring:
                        break
                    print("Will retry in the next update cycle...")
                    time.sleep(UPDATE_INTERVAL)

            # Ask if user wants to process another state/year
            another = input("\nWould you like to process another state/year? (yes/no): ").strip().lower()
            if another not in ['yes','y']:
                break

        except KeyboardInterrupt:
            print("\nProgram terminated by user")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            break

    print("\nProgram completed. Thank you for using LegiScan Bill Tracker!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback

        traceback.print_exc()