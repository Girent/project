import time

from stats import predictRisks
import json
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/risks', methods=['POST'])
def handle_post_request():
    data = request.json

    if data is None:
        return jsonify({"error": "Incorrect structure data"}), 400

    result = predictRisks(data["data"])

    try:
        result_json = json.loads(result)
    except json.JSONDecodeError:
        result_json = None

    output_data = {
        "data": data["data"],
        "result": result_json,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open("output.json", "r") as f:
            existing_data = json.load(f)

    except FileNotFoundError:
        existing_data = []

    existing_data.append(output_data)

    with open("output.json", "w") as f:
        json.dump(existing_data, f)

    return jsonify(output_data)


@app.route('/risks', methods=['GET'])
def get_risks():
    try:
        with open("output.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
