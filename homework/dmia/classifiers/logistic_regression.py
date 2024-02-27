import numpy as np
from scipy import sparse


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this classifier using stochastic gradient descent.

        Inputs:
        - X: N x D array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # Add a column of ones to X for the bias sake.
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01  # (dim, )

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in range(num_iters):
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            random_indices = np.random.choice(num_train, batch_size)
            X_batch = X[random_indices]
            y_batch = y[random_indices]
            
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w -= learning_rate * gradW

            
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return self

    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z)) 
    
    def predict_proba(self, X, append_bias=False):
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of data. Each row is a D-dimensional point. !(maybe N = batch size)!
        - append_bias: bool. Whether to append bias before predicting or not.

        Returns:
        - y_proba: Probabilities of classes for the data in X. y_pred is a 2-dimensional
          array with a shape (N, 2), and each row is a distribution of classes [prob_class_0, prob_class_1].
        """
        if append_bias:
            X = LogisticRegression.append_biases(X)
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the probabilities of classes in y_proba.   #
        # Hint: It might be helpful to use np.vstack and np.sum                   #
        ###########################################################################
        # prob_y1 = 1/(1 + exp(-XB)), где XB = X1B1 + X2B2...XnBn
        print(X.shape)
        print(self.w.shape)
#         prob_class_1 = 1/(1 + np.exp(-1*XB_sum))  # (N,)

        prob_class_1 = self.sigmoid(np.dot(X, self.w))
        prob_class_0 = 1 - prob_class_1

        prob_class_1 = prob_class_1.reshape(-1,1) # (N, 1)
        prob_class_0 = prob_class_0.reshape(-1,1)

        y_proba = np.concatenate((prob_class_0, prob_class_1), axis=1) # (N, 2)
        
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_proba

    def predict(self, X):
        """
        Use the ```predict_proba``` method to predict labels for data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        y_proba = self.predict_proba(X, append_bias=True)
        y_pred = y_proba.copy()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = y_pred[:,0] # (N,)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    
    def get_loss_sum(y, y_pred):
    
        # рассчитаем функцию потерь для y = 1, добавив 1e-9, чтобы избежать ошибки при log(0)
        y_one_loss = y * np.log(y_pred + 1e-9)

        # также рассчитаем функцию потерь для y = 0
        y_zero_loss = (1 - y) * np.log(1 - y_pred + 1e-9)

        # сложим и разделим на количество наблюдений
        return -np.sum(y_zero_loss + y_one_loss)
        
    
    
    def get_gradient_sum(x, y, y_pred, n):
        return np.dot(x.T, (y_pred - y))
    
    
    def loss(self, X_batch, y_batch, reg):
        """Logistic Regression loss function
        Inputs:
        - X: N x D array of data. Data are D-dimensional rows
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes
        Returns:
        a tuple of:
        - loss as single float
        - gradient with respect to weights w; an array of same shape as w
        """
        dw = np.zeros_like(self.w)  # initialize the gradient as zero
        N = X_batch.shape[0]
        loss = 0
        # Compute loss and gradient. Your code should not contain python loops.
        # *Произвести предсказания и подсчитать ошибки
        
        y_pred = self.predict(X_batch)
        
        loss_sum = get_loss(y_batch, y_pred)
        
        gradient_sum = get_gradient(X_batch, y_batch, y_pred, N)
        

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        # Note that the same thing must be done with gradient.
        loss = loss_sum / N
        dw = gradient_sum / N

        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.
        theta = np.zeros(X_batch.shape[1])  # Initialize parameter vector
        
        regularization_term_loss = (lambda_reg / (2 * m)) * np.sum(np.square(theta[1:]))  # excluding bias term
        regularization_term_grad = (lambda_reg / m) * np.concatenate(([0], theta[1:]))  # excluding bias term
        
        loss += regularization_term_loss
        dw += regularization_term_grad
        
        return loss, dw

    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    