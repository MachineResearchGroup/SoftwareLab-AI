#!bin/bash
export FLASK_APP=handler.py;
export FLASK_ENV=development;
cd ./src/service;
flask run;
