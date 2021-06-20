""" Script with the model definiton """

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os
from glob import glob
from time import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.models import load_model

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

class Model(object):
    '''
    Model class definition
    '''
    def __init__(self, input_shape, path_save):
        """
        Initializer method for the model.
        
        Parameters
        ----------
        input_shape : tuple
            Input shape for the model
        path_save : string
            Path to save model checkpoints.
        """
        
        self.path_save = path_save
        self.input_shape = input_shape
        
        # placeholder for trained models with KFolds
        self.trained_models = []
    
    def __get_model(self):
        """ Define a simple sequential fully connected NN """
        
        # define model
        model = tf.keras.Sequential([
            KL.Dense(128, input_shape=self.input_shape),
            KL.PReLU(),
            KL.Dense(256),
            KL.PReLU(),
            KL.Dense(128),
            KL.PReLU(),
            KL.Dense(1, activation='tanh')
        ])
        # compile model
        model.compile(tf.keras.optimizers.Adam(learning_rate=3e-3), loss='huber')
        
        return model
            
    def train(self,
              X: np.array,
              y: np.array,
              batch_size: int=64,
              epochs: int=100,
              verbose: bool=False,
              n_splits: int=10):
        """
        Train the model.
        
        Parameters
        ----------
        X : numpy.array
            Input data
        y : numpy.array
            Target data
        batch_size : int (optional, default=64)
            Batch size to be used during training
        epochs : int (optional, default=64)
            Number of epochs that each model will be trained
        verbose : bool (optinal, default=False)
            Verbose level for the training.
        n_splits : int (optional, default=10)
            Number of splits in the KFold algorithm. The number of models to be trained is related to this value.
        """
        
        print('Training model...')
        
        # KFold algorithm
        skf = KFold(n_splits=n_splits)
        skf.get_n_splits(X, y)
    
        # train the model for each fold
        scores = []
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            init = time()
            
            # split data between training and testing
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # get a new model
            model = self.__get_model()
            
            # fit the model with the data
            model.fit(
                    x=X_train,
                    y=y_train,
                    shuffle=True,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks = [
                        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
                        tf.keras.callbacks.ModelCheckpoint(os.path.join(self.path_save, f'nn_model_{i}.h5'), save_best_only=True),
                    ],
                    validation_data = (X_test, y_test),
                    verbose=0
                )
            
            # load best weights
            model.load_weights(os.path.join(self.path_save, f'nn_model_{i}.h5'))
            
            # evaluate the model on the R2 score
            pred = model.predict(X_test)
            scores.append(r2_score(y_test, pred))
            
            if verbose:
                print(f'\nIteration {i+1}/{n_splits}')
                print('R2 score: ', scores[-1])
                print('Elapsed time: ', time()-init, 'sec')
                print('-'*15)
            del model
        
        if verbose:
            print('Mean R2 score: ', np.mean(scores))
            
    def load_models(self, path_to_saves):
        """
        Load trained models.
        
        Parameters
        ----------
        path_to_saves : string
            Path to the folder with the trained models
        """
        
        print('Loading models...')
        
        # load all trained models in the folder
        path_models = glob(os.path.join(path_to_saves, '*.h5'))
        assert len(path_models)>0, f'Found no trained model in the provided path: {path_to_save}'
        
        self.trained_models = [load_model(path) for path in path_models]
        
    def predict(self, x):
        """
        Return the mean prediction to all trained models.
        
        Parameters
        ----------
        x : np.array
            Input data
        """
        
        assert len(self.trained_models)>0, 'No recorded trained model. Call load_model method before predict.'
        
        print('Predicting data...')
        
        # predict data in all trained models and return the mean
        pred = [m.predict(x) for m in self.trained_models]
        return np.mean(pred, axis=0)