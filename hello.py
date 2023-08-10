from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return '', 200

if __name__ == '__main__':
    app.run()
