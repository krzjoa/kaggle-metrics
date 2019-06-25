# kaggle-metrics <img src="https://raw.githubusercontent.com/krzjoa/kaggle-metrics/master/img/kmlogo.png" align="right" width = "120px"/>

Metrics for Kaggle competitions.

**See wiki**: [Implemented metrics](https://github.com/krzjoa/kaggle-metrics/wiki/Implemented-metrics)

## Installation
```bash
sudo pip install git+https://github.com/krzjoa/kaggle-metrics.git
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


