from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib as plt

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)
            
    def __init__(self, regularization, lr=0.01, method='batch', init='xavier', polynomial=True, degree=2, use_momentum=True, momentum=0.5, num_epochs=500, batch_size=50, cv=kfold):
    

        # Initialize hyperparameters and options
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.polynomial = polynomial
        self.degree     = degree
        self.init       = init
        self.use_momentum   = use_momentum
        self.momentum   = momentum
        self.prev_step  = 0
        self.cv         = cv
        self.regularization = regularization

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def r2(self, ytrue, ypred):
        return 1 - (((ytrue - ypred) ** 2).sum() / ((ytrue - ytrue.mean()) ** 2).sum())

    # function to compute average mse for all kfold_scores
    def avgMse(self):
        return np.sum(np.array(self.kfold_scores))/len(self.kfold_scores)
    
    # function to compute average r2 for all kfold_scores
    def avgr2(self):
        return np.sum(np.array(self.kfold_r2))/len(self.kfold_r2)
    
    def fit(self, X_train, y_train):
    
        # Store column names for later use

        self.columns = X_train.columns

        if self.polynomial == True:
            X_train = self._transform_features(X_train)
            print("Using Polynomial")
        else:
            print("Using Linear")
            X_train = X_train.to_numpy()
            
        y_train = y_train.to_numpy()
        
        #create a list of kfold scores
        self.kfold_scores = list()
        self.kfold_r2 = list()
        
        #reset val loss
        self.val_loss_old = np.inf

        #X_train = np.c_[np.ones((X_train.shape[0],1)), X_train]
        # Perform k-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            #initialize weights using Xavier method
            if self.init == 'xavier':
                #calculate the range for the weights with number of samples
                n_features = X_cross_train.shape[1]
                lower, upper = -(1 / np.sqrt(n_features)), 1 / np.sqrt(n_features)
                #randomize weights then scale them using lower and upper bounds
                self.theta = np.random.rand(X_cross_train.shape[1])
                self.theta = lower + self.theta * (upper - lower)

            #initialize weights with zero
            elif self.init == 'zero':
                self.theta = np.zeros(X_cross_train.shape[1])

            else:
                print("Wrong weights init method. Must be either 'xavier' or 'zero'")
                return
            # Start training for the current fold
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {
                    "method": self.method,
                    "lr": self.lr,
                    "reg": type(self).__name__
                }
                
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]  
                    y_cross_train = y_cross_train[perm]
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]): 
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1)
                            y_method_train = y_cross_train[batch_idx].reshape(1, ) 
                            train_loss = self._train(X_method_train, y_method_train) 
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)
                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
                    yhat_val = self._predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    val_r2_new = self.r2(y_cross_val, yhat_val)                    
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    mlflow.log_metric(key="val_r2", value=val_r2_new, step=epoch)

                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
                self.kfold_scores.append(val_loss_new)
                self.kfold_r2.append(val_r2_new)
                
                print(f"Fold {fold}: MSE {val_loss_new}")
                print(f"Fold {fold}: R2: {val_r2_new}")

    def _transform_features(self, X):
        X = X.to_numpy() if hasattr(X, 'to_numpy') else X
        features = [np.ones((X.shape[0], 1))]  # Bias term
        feature_names = ["bias"]
        orig_cols = self.columns if hasattr(self, 'columns') and self.columns is not None else [f"x{i}" for i in range(X.shape[1])]
    
        # Adding original features
        features.append(X)
        feature_names.extend(orig_cols)

        # Adding polynomial features
        for d in range(2, self.degree + 1):
            features.append(X ** d)
            feature_names.extend([f"{col}^{d}" for col in orig_cols])

        X_poly = np.hstack(features)
        self.poly_feature_names = feature_names  
        return X_poly
            
                    
    def _train(self, X, y):
        yhat = self._predict(X)
        m    = X.shape[0]    
        if self.regularization:   
            # If regularization is enabled, compute the gradient with regularization term 
            grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        else:
            # If no regularization, compute the gradient without regularization term
            grad = (1/m) * X.T @(yhat - y)

        if self.use_momentum == True:
            self.step = self.lr * grad
            self.theta = self.theta - self.step + self.momentum * self.prev_step
            self.prev_step = self.step
        else:
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)
    
    def _predict(self, X):
        return X @ self.theta   # Matrix multiplication of input features and model weights to make predictions
    
    def predict(self, X):
        if self.polynomial == True:
            X = self._transform_features(X)
        return X @ self.theta  # Matrix multiplication of input features and model weights to make predictions
    
    def _coef(self):
        return self.theta[1:] # Return all weights except the first one (bias term)
    def _bias(self):
        return self.theta[0] # Return the first weight, which represents the bias term

    def feature_importance(self, width=5, height=10):
        if self.theta is not None:
            index_names = getattr(self, 'poly_feature_names', 
                             self.columns if hasattr(self, 'columns') and self.columns is not None else None)
            if index_names is None:
                print("Feature names are not available.")
                return

            if len(self.theta) != len(index_names):
                raise ValueError(
                    f"Mismatch: theta has {len(self.theta)} elements, "
                    f"but feature names has {len(index_names)} names."
                )

            coefs = pd.DataFrame(data=self.theta, columns=['Coefficients'], index=index_names)
            coefs.plot(kind="barh", figsize=(width, height))
            plt.title("Feature Importance")
            plt.show()
        else:
            print("Coefficients are not available to create the graph.")

class LassoPenalty:
    def __init__(self, l):
        self.l = l  # Lambda value for regularization
        
    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)

# Define a class for Ridge (L2) regularization penalty
class RidgePenalty:
    def __init__(self, l):
        self.l = l  # Lambda value for regularization
        
    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta

# Define a class for Elastic Net regularization penalty
class ElasticPenalty:
    def __init__(self, l=0.1, l_ratio=0.5):
        self.l = l  # Total regularization strength
        self.l_ratio = l_ratio  # Ratio between L1 and L2 regularization
        
    def __call__(self, theta):
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)

# Define classes for Lasso, Ridge, ElasticNet, and Normal (No Regularization) linear regression
class Lasso(LinearRegression):
    def __init__(self, l, lr, method, init, polynomial, degree, use_momentum, momentum):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method, init, polynomial, degree, use_momentum, momentum)
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

class Ridge(LinearRegression):
    def __init__(self, l, lr, method, init, polynomial, degree, use_momentum, momentum):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, init, polynomial, degree, use_momentum, momentum)
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

class ElasticNet(LinearRegression):
    def __init__(self, l, lr, method, init, polynomial, degree, use_momentum, momentum, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method, init, polynomial, degree, use_momentum, momentum)
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

class Normal(LinearRegression):  
    def __init__(self, l, lr, method, init, polynomial, degree, use_momentum, momentum):
        self.regularization = None  # No regularization
        super().__init__(self.regularization, lr, method, init, polynomial, degree, use_momentum, momentum)
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

