from flask import Flask

app= Flask(__name__)

@app.route('/')
def DALYS():
    return 'Hey'

app.run(debug=True)