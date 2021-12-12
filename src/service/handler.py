from infra.ioc.injector import DependencyInjector
from service.mapper import Mapper
from flask import (Flask, request)
import logging

app = Flask("SoftwareLab-AI")
logging.basicConfig(
    level=logging.DEBUG,
    format=f"\n[%(asctime)s] %(levelname)s in module %(module)s, "
    f"function %(funcName)s and line %(lineno)d: \n%(message)s\n"
)

classification_app = DependencyInjector().classification_app


@app.route("/", methods=['POST', "GET"])
def post():
    data = request.get_data(as_text=True)
    logging.info(f"Requirement received: {data}")
    requirement = Mapper.to_requirement(data=data)
    response = classification_app.run(requirement=requirement)
    logging.info(f"Response: {response}")
    return response


@app.route("/setting", methods=["POST"])
def setting():
    """"""
