from setuptools import setup
setup(
  name = 'kaggle-metrics',
  packages = ['kaggle_metrics'], # this must be the same as the name above
  version = '0.1',
  description = 'Metrics for Kaggle competitions',
  author = 'Krzysztof Joachimiak',
  author_email = 'joachimiak.krzysztof@gmail.com',
  url = 'https://github.com/krzjoa/kaggle-metrics', # use the URL to the github repo
 # download_url = 'https://github.com/krzjoa/kaggle-metrics', # I'll explain this in a second
  keywords = ['kaggle', 'metrics'],
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Topic :: Utilities",
  ]
)