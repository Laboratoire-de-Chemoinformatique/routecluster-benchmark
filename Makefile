PYTHON_VERSION ?= 3.11
UV ?= uv
VENV_MAIN ?= .venv
VENV_AB ?= .venv-ab
VENV_TED ?= .venv-ted

.PHONY: uv-main uv-ab uv-ted

uv-main:
	$(UV) sync --python $(PYTHON_VERSION) --venv $(VENV_MAIN)

uv-ab:
	$(UV) sync --extra ab --python $(PYTHON_VERSION) --venv $(VENV_AB)

uv-ted:
	$(UV) sync --extra ted --python $(PYTHON_VERSION) --venv $(VENV_TED)
