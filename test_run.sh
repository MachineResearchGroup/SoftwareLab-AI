cd ./src/test/;
python -m unittest;
coverage html ./test*.py;
coverage report ./test*.py > out_coverage_tests.txt;
