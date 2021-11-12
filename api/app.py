import werkzeug
from flask.scaffold import _endpoint_from_view_func
from werkzeug.utils import cached_property
import flask

flask.helpers._endpoint_from_view_func = _endpoint_from_view_func

werkzeug.cached_property = cached_property

from flask import Flask
from flask import request
from flask_restplus import Resource, Api
from flask_restplus import reqparse
import json
import os
import pickle
import numpy as np
import base64

app = Flask(__name__)
api = Api(app)

@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

@api.route('/check/<number>', endpoint="/check")
class Check(Resource):
    def get(self, number):
        return {'Is even : ': str( 'No' if (int(number) % 2 > 0) else 'Yes' ) }

'''
@api.route('/predict', method=['POST'])
def predict():
    inputJson = request.json
    print(inputJson.image)
'''

loaded_model = pickle.load(open('../model/model_svm_0.0001.sav', 'rb'))

@api.route('/predict')
class Predict(Resource):
    def post(self):
        inputJson = request.json
        image = inputJson['image']

        image = np.array(image).reshape(1, -1)
        predicted = loaded_model.predict(image)

        return str(predicted[0])

if __name__ == '__main__':
    app.run(debug=True)