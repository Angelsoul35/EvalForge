.PHONY: install test lint run gate calibrate demo clean

install:
	pip install -e ".[dev,api]"

test:
	pytest --cov=evalforge --cov-report=term-missing

lint:
	ruff check evalforge tests

run:            ## Run the example suite and render a report
	evalforge run examples/rag_suite.yaml --out reports/report.json

gate:           ## CI quality gate on the example suite (thresholds suit the offline judge)
	evalforge gate examples/rag_suite.yaml --min-hit-rate 0.70 --min-faithfulness 0.40

calibrate:      ## Compare the judge against human labels
	evalforge calibrate examples/rag_suite.yaml --labels examples/human_labels.json

demo: run gate calibrate

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov reports/*.json
	touch reports/.gitkeep
