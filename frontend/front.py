# front.py
from flask import Flask, request, jsonify # type: ignore
import subprocess
import os
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    # Run the chatbot script and capture output
    result = subprocess.run(
        ['python', 'chatbot.py'],
        input=user_input,
        text=True,
        capture_output=True
    )
    return jsonify({'response': result.stdout.strip()})  # Strip extra whitespace

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))