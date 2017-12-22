[km]: https://raw.githubusercontent.com/krzjoa/kaggle-metrics/master/km50p.png "kaggle-metrics logo" 

![alt text][km]  

# kaggle-metrics


Metrics for Kaggle competitions.

List: https://www.kaggle.com/wiki/Metrics/history/50647

## Implemented metrics

### Regression metrics

* Mean absolute error
* Weighted mean absolute error
* Root mean squared error
* Root mean squared logarithmic error

### Classification metrics

* Logarithmic loss
* Mean consequential error
* Hamming loss
* Mean utility
* Matthews Correlation Coefficient

### Order metrics
TODO

### Metrics for probability distribution function
TODO

### Error Metrics for Retrieval Problems
TODO

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


