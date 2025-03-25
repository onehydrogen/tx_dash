from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World! Flask is now working!'

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
