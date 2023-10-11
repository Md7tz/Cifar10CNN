default: help
.PHONY: install format lint clean

help:
	@echo "install - install dependencies"
	@echo "format - format code with black"
	@echo "lint - lint code with flake8"

install: requirements.txt
	pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 *.py

clean:
	rm -rf __pycache__
	rm -rf data