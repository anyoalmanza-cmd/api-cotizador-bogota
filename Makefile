.PHONY: train test run clean

        train:
        	python3 -m src.train

        test:
        	pytest tests/

        run:
        	uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

        clean:
        	find . -type d -name "__pycache__" -exec rm -rf {} +
        	find . -type d -name ".pytest_cache" -exec rm -rf {} +
