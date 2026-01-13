import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

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
    with open(f"scenes/{scene_name}.json") as f:
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

if __name__ == "__main__":
    app.run(debug=True)
