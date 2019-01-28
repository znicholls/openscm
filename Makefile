DATA_DIR = ./data


RCPS_DIR=$(DATA_DIR)/rcps
RCPHISTORICAL_EMISSIONS=$(RCPS_DIR)/20THCENTURY_EMISSIONS.DAT
RCP26_EMISSIONS=$(RCPS_DIR)/RCP26_EMISSIONS.DAT
RCP45_EMISSIONS=$(RCPS_DIR)/RCP45_EMISSIONS.DAT
RCP60_EMISSIONS=$(RCPS_DIR)/RCP60_EMISSIONS.DAT
RCP85_EMISSIONS=$(RCPS_DIR)/RCP85_EMISSIONS.DAT
RCPS_EMISSIONS=$(RCPHISTORICAL_EMISSIONS) $(RCP26_EMISSIONS) $(RCP45_EMISSIONS) $(RCP60_EMISSIONS) $(RCP85_EMISSIONS)


.PHONY: full-dev-setup
full-dev-setup: venv $(RCPS_EMISSIONS)

rcps-data: $(RCPS_EMISSIONS)

$(RCPHISTORICAL_EMISSIONS):
	mkdir -p $(RCPS_DIR)
	wget http://www.pik-potsdam.de/~mmalte/rcps/data/20THCENTURY_EMISSIONS.DAT -O $@
	touch $@

$(RCP26_EMISSIONS):
	mkdir -p $(RCPS_DIR)
	wget http://www.pik-potsdam.de/~mmalte/rcps/data/RCP3PD_EMISSIONS.DAT -O $@
	touch $@

$(RCP45_EMISSIONS):
	mkdir -p $(RCPS_DIR)
	wget http://www.pik-potsdam.de/~mmalte/rcps/data/RCP45_EMISSIONS.DAT -O $@
	touch $@

$(RCP60_EMISSIONS):
	mkdir -p $(RCPS_DIR)
	wget http://www.pik-potsdam.de/~mmalte/rcps/data/RCP6_EMISSIONS.DAT -O $@
	touch $@

$(RCP85_EMISSIONS):
	mkdir -p $(RCPS_DIR)
	wget http://www.pik-potsdam.de/~mmalte/rcps/data/RCP85_EMISSIONS.DAT -O $@
	touch $@

venv: setup.py
	[ -d ./venv ] || python3 -m venv ./venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e .[tests,docs,dev]
	touch venv

test: venv
	./venv/bin/pytest -rfsxEX --cov=openscm tests

coverage: test
	coverage html

test_all: test venv
	./venv/bin/pytest -rfsxEX --nbval ./notebooks --sanitize ./notebooks/tests_sanitize.cfg

docs: venv
	./venv/bin/sphinx-build -M html docs docs/build

flake8: venv
	./venv/bin/flake8 openscm tests

black: venv
	@status=$$(git status --porcelain openscm tests); \
	if test "x$${status}" = x; then \
		./venv/bin/black --exclude _version.py setup.py openscm tests; \
	else \
		echo Not trying any formatting. Working directory is dirty ... >&2; \
	fi;

publish-on-pypi: venv
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		./venv/bin/python setup.py bdist_wheel --universal; \
		./venv/bin/twine upload dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-pypi-install: venv
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install openscm
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import openscm; print(openscm.__version__)"

clean:
	rm -rf venv

.PHONY: clean coverage test test-all black flake8 docs publish-on-pypi test-pypi-install
