from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

modelo = joblib.load('modelo_extra_trees.pkl')

@app.route('/prever', methods=['POST'])
def prever():
    
    if not request.json:
        return jsonify({'error': 'Requisição inválida. JSON esperado.'}), 400
    
    
    if 'Close' not in request.json or 'Volume' not in request.json:
        return jsonify({'error': 'Os campos Close e Volume são obrigatórios.'}), 400
    
    try:
        valor_Close = float(request.json['Close'])
        valor_Volume = float(request.json['Volume'])
    except ValueError:
        return jsonify({'error': 'Os valores de Close e Volume devem ser floats.'}), 400
    
    instance = np.array([valor_Close, valor_Volume])
    prediction = modelo.predict([instance])[0]  
    
    return jsonify({'resultado': prediction}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
