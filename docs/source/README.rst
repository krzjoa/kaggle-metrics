kaggle-metrics
==============
.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: https://bace.readthedocs.io/en/latest/?badge=latest
    :alt: Python version
.. image:: https://badge.fury.io/py/kaggle-metrics.svg
    :target: https://badge.fury.io/py/kaggle-metrics
    :alt: PyPI version
.. image:: https://readthedocs.org/projects/kaggle-metrics/badge/?version=latest
    :target: https://kaggle-metrics.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT)
    :alt: Licence

Metrics for Kaggle competitions.

Installation
------------

You can install this module directly from GitHub repo with command:

.. code-block::

   python3.7 -m pip install git+https://github.com/krzjoa/kaggle-metrics.git

or as a PyPI package

.. code-block::

   python3.7 -m pip install kaggle_metrics

Usage
-----
.. code-block:: python

    from xgboost import XGBRegressor
    import kaggle_metrics as km

    X_train, y_train, X_test, y_test = get_data()

    # Train
    clf = XGBRegressor()
    clf.fit(X_train, y_train)

    # Get predictions
    y_pred = clf.predict(X_test)

    # Evaluate with kaggle-metrics
    km.rmse(y_test, y_pred)
