from flask import Flask, request, jsonify
from evaluate import evaluate_response
import json
import os

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.get_json()
        response = data.get('response')
        question = data.get('question')
        
        if not response or not question:
            return jsonify({'error': 'Missing response or question'}), 400
            
        evaluation = evaluate_response(response, question)
        return jsonify(evaluation)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)