from setuptools import setup

# python3 setup.py sdist bdist_wheel
# python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

readme = open('README.md').read()
doclink = """
Documentation
-------------
The full documentation is at http://kaggle-metrics.rtfd.org."""

VERSION = '0.2.0'


setup(
  name = 'kaggle-metrics',
  packages = ['kaggle_metrics'], #
  version = VERSION,
  description = 'Metrics for Kaggle competitions',
  long_description=readme + '\n\n' + doclink + '\n\n',
  author = 'Krzysztof Joachimiak',
  author_email = 'joachimiak.krzysztof@gmail.com',
  url = 'https://github.com/krzjoa/kaggle-metrics',
  long_description_content_type="text/markdown",
 # download_url = 'https://github.com/krzjoa/kaggle-metrics',
  keywords = ['kaggle', 'metrics'],
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Topic :: Utilities",
  ]
)