from application.classification_app import ClassificationApp
from service.mapper import Mapper
from flask import (Flask, request)
app = Flask("SoftwareLab-AI")


@app.route("/", methods=['POST'])
def post():
    data = request.get_data(as_text=True)
    app.logger.info(f"\n\nRequirement received: {data}\n")
    requirement = Mapper.to_requirement(data=data)
    response = ClassificationApp().run(requirement=requirement)
    app.logger.info(f"\nResponse: {response}\n")
    return response


@app.route("/setting", methods=["POST"])
def setting():
    """"""
