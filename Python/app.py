from flask import Flask, render_template,request,jsonify
from utils import predict_fertilizer
from flask_cors import CORS

app=Flask(__name__)

CORS(app)


@app.route('/predict',methods=['POST'])
def predict():
    data=request.json
    inputs=data.get('inputs')
    output=predict_fertilizer(inputs)
    return jsonify({'prediction':output})


if __name__=="__main__":
    app.run(debug=True)