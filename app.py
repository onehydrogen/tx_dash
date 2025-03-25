from flask import Flask
import dash
import dash_bootstrap_components as dbc
from dash import html

# Initialize the Flask server
server = Flask(__name__)

# Initialize the Dash app with the Flask server
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.FLATLY]
)

# Create a simple layout
app.layout = html.Div([
    html.H1('Texas Legislative Dashboard'),
    html.P('Basic version working on Heroku')
])

# Run the app
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run_server(host='0.0.0.0', port=port, debug=False)
