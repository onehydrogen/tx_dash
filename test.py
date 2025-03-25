import flask
import dash
import dash_bootstrap_components as dbc
from dash import html

# Create a simple Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # This is important for Heroku

app.layout = html.Div([
    html.H1("Test App for Heroku"),
    html.P("If you can see this, the deployment is working!")
])

if __name__ == '__main__':
    app.run_server(debug=True)
