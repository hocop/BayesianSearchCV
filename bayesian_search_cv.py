from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

class BayesianSearchCV:
    '''
    Bayesian Search with cross validation score.
    
    Arguments:
    
    base_estimator: sklearn-like model
    param_bounds: dict
        hyperparameter upper and lower bounds
        example: {
            'param1': [0, 10],
            'param2': [-1, 2],
        }
    scoring: string or callable
        scoring argument for cross_val_score
    cv: int
        number of folds
    n_iter: int
        number of bayesian optimization iterations
    init_points: int
        number of random iterations before bayesian optimization
    random_state: int
        random_state for bayesian optimization
    int_parameters: list
        list of parameters which are required to be of integer type
        example: ['param1', 'param3']
    '''
    
    def __init__(
        self,
        base_estimator,
        param_bounds,
        scoring,
        cv=5,
        n_iter=50,
        init_points=10,
        random_state=1,
        int_parameters=[],
    ):
        self.base_estimator = base_estimator
        self.param_bounds = param_bounds
        self.cv = cv
        self.n_iter = n_iter
        self.init_points = init_points
        self.scoring = scoring
        self.random_state = random_state
        self.int_parameters = int_parameters
    
    def objective(self, **params):
        '''
        We will aim to maximize this function
        '''
        # Turn some parameters into ints
        for key in self.int_parameters:
            if key in params:
                params[key] = int(params[key])
        # Set hyperparameters
        self.base_estimator.set_params(**params)
        # Calculate the cross validation score
        cv_scores = cross_val_score(
            self.base_estimator,
            self.X_data,
            self.y_data,
            cv=self.cv,
            scoring=self.scoring)
        score = cv_scores.mean()
        return score
    
    def fit(self, X, y):
        self.X_data = X
        self.y_data = y
        
        # Create the optimizer
        self.optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=self.param_bounds,
            random_state=self.random_state,
        )
        
        # The optimization itself goes here:
        self.optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )
        
        del self.X_data
        del self.y_data
        
        # Save best score and best model
        self.best_score_ = self.optimizer.max['target']
        self.best_params_ = self.optimizer.max['params']
        for key in self.int_parameters:
            if key in self.best_params_:
                self.best_params_[key] = int(self.best_params_[key])
        
        self.best_estimator_ = clone(self.base_estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        
        return self
    
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)
