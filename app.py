from flask import Flask
import dash
import dash_bootstrap_components as dbc
from dash import html, dash_table
import pandas as pd

# Initialize the Flask server
server = Flask(__name__)

# Initialize the Dash app with the Flask server
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.FLATLY]
)

# Create sample data
def get_sample_data():
    return pd.DataFrame({
        'year': ['2023', '2023'],
        'bill_number': ['HB1234', 'SB5678'],
        'title': ['Sample Bill 1', 'Sample Bill 2'],
        'status': ['active', 'in_process'],
        'stage': ['prefiled', 'in_committee']
    })

# Get data
df = get_sample_data()

# Create a simple layout
app.layout = html.Div([
    html.H1('Texas Legislative Dashboard', style={'textAlign': 'center'}),
    html.P('Basic version working on Heroku', style={'textAlign': 'center'}),
    html.Hr(),
    dash_table.DataTable(
        id='bills-table',
        columns=[{'name': col.title(), 'id': col} for col in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px'
        },
        style_header={
            'backgroundColor': '#00205B',
            'color': 'white',
            'fontWeight': 'bold'
        }
    )
], style={'padding': '20px'})

# Run the app
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run_server(host='0.0.0.0', port=port, debug=False)
