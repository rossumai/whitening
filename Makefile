install:
	pip install whitening

install_dev:
	pip install -e .[test]

uninstall:
	pip uninstall whitening

test:
	python -m pytest tests/

clean:
	rm -r build/ dist/ whitening.egg-info/

# twine - a tool for uploading packages to PyPI
install_twine:
	pip install twine

build:
	python setup.py sdist
	python setup.py bdist_wheel --universal

# PyPI production

publish:
	twine upload dist/*
