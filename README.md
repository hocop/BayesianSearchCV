# BayesianSearchCV
`BayesianSearchCV` for sklearn models.

## Usage

This class is similar to `GridSearchCV` and `RandomizedSearchCV` from `sklearn.model_selection`:

```
import lightgbm as lgb
from bayesian_search_cv import BayesianSearchCV

model = lgb.LGBMClassifier(n_estimators=200, num_leaves=40)

# Set only upper and lower bounds for each parameter
param_grid = {
    'learning_rate': (0.01, 1),
    'n_estimators': (10, 200),
    'num_leaves': (2, 50),
    'reg_alpha': (0, 1),
    'reg_lambda': (0, 10),
    'min_split_gain': (0, 1),
}

model_bayes = BayesianSearchCV(
    model, param_grid, cv=5, n_iter=60, scoring='accuracy',
    int_parameters=['n_estimators', 'num_leaves'])

model_bayes.fit(X_train, y_train)
```

Full usage example on the Titanic dataset: https://www.kaggle.com/hocop1/3-approaches-to-hyperparameter-search-bayesian  

`BayesianSearchCV` class is imported from `bayesian_search_cv.py` file.  
This is one-file repository, so no need to install it. Just copy this file into your working directory.

## Requirements
This class is based on [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)  
You may need to install it:
```
$ pip install bayesian-optimization
```
