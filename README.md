# kaggle-metrics <img src="https://raw.githubusercontent.com/krzjoa/kaggle-metrics/master/img/kmlogo.png" align="right" width = "120px"/>
![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg) 
[![PyPI version](https://badge.fury.io/py/kaggle-metrics.svg)](https://badge.fury.io/py/kaggle-metrics ) 
[![Documentation Status](https://readthedocs.org/projects/kaggle-metrics/badge/?version=latest)](https://kaggle-metrics.readthedocs.io/en/latest/?badge=latest) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Metrics for Kaggle competitions.

**See wiki**: [Implemented metrics](https://github.com/krzjoa/kaggle-metrics/wiki/Implemented-metrics)

## Installation
```bash
python3.7 -m pip install git+https://github.com/krzjoa/kaggle-metrics.git
```
or:

```bash
python3.7 -m pip install kaggle_metrics
```

## Usage
```python
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


```


