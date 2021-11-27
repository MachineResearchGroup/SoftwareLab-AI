import logging
from flask import Flask
import logging
from application.classification_app import ClassificationApp

app = Flask(__name__)


@app.route("/", methods='POST')
def post(requirements: dict):
    logging.info(f"Requirements:\n {requirements}")
    return ClassificationApp().run(requirements=requirements)
