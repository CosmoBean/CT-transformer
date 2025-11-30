.PHONY: install clean

install:
	bash scripts/install.sh

clean:
	rm -rf .venv uv.lock main.py pyproject.toml
	rm -rf cache