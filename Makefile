PY ?= python
.PHONY: test web-data web-build
test:  ## CPU unit tests (each prints ALL_PASS / *_TEST_PASS)
	$(PY) tests/test_v2_training_path.py
	$(PY) tests/test_analysis.py
	$(PY) tests/test_human_eval.py
	$(PY) tests/test_independent_eval.py
	$(PY) scripts/estimator_checks.py
web-data:  ## regenerate web/public/data/results.json from results/
	$(PY) scripts/build_web_data.py
web-build: web-data
	cd web && npm ci && npm run build
