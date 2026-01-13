import json
import logging
import webbrowser
import threading

from flask import Flask, request, jsonify, render_template

logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)
app.logger.setLevel(logging.ERROR)

STATE = {
    "scene": None,
    "dialogue": None,
    "speaker": None
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scene", methods=["POST"])
def load_scene():
    scene_name = request.json["scene"]
    with open(f"ui/scenes/{scene_name}.json") as f:
        STATE["scene"] = json.load(f)
        STATE["dialogue"] = None
        STATE["speaker"] = None
    return {"ok": True}

@app.route("/say", methods=["POST"])
def say():
    data = request.json

    STATE["dialogue"] = data["text"]
    STATE["speaker"] = data["who"]
    print(STATE)
    return {"ok": True}

@app.route("/state")
def state():
    return jsonify(STATE)

def open_browser():
    webbrowser.open_new_tab("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(0.5, open_browser).start()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )
