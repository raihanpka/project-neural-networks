import numpy as np

class BaseLoss:
    # Metode untuk menghitung data loss
    def calculate(self, output, y, layer=None):
        # Menghitung sample_losses berdasarkan output dan y
        sample_losses = self.forward(output, y, layer)
        # Menghitung data_loss dengan menggunakan np.mean
        data_loss = np.mean(sample_losses)
        return data_loss
    
    # Metode abstrak untuk menghitung sample_losses
    def forward(self, y_pred, y_true, layer=None):
        raise NotImplementedError
    
    # Metode abstrak untuk menghitung gradient
    def backward(self, y_pred, y_true, layer=None):
        raise NotImplementedError

class MeanSquaredError(BaseLoss):
    def __init__(self, regularization_l2=0.0, regularization_l1=0.0):
        self.regularization_l2 = regularization_l2
        self.regularization_l1 = regularization_l1

    def forward(self, y_pred, y_true, layer=None):
        # Forward for MSE: support both integer labels (1D) and full targets (2D)
        y_pred = np.array(y_pred)
        sample_size = len(y_pred)

        # Convert integer labels to one-hot if necessary (only for multi-class classification)
        if len(y_true.shape) == 1 and y_pred.ndim > 1 and y_pred.shape[1] > 1:
            n_outputs = y_pred.shape[1]
            y_true_ohe = np.zeros_like(y_pred)
            y_true_int = y_true.astype(int)
            y_true_ohe[range(sample_size), y_true_int] = 1
        else:
            y_true_ohe = y_true

        # compute MSE per element and average
        squared_error = np.square(y_pred - y_true_ohe)
        data_loss = np.mean(squared_error)

        # regularization
        regularization_loss = self._calculate_regularization_loss(layer)
        return data_loss + regularization_loss

    def backward(self, y_pred, y_true, layer=None):
        # Backward for MSE
        y_pred = np.array(y_pred)
        sample_size = len(y_pred)

        if len(y_pred.shape) == 1:
            # if predictions are 1D vectors
            y_pred = y_pred.reshape(-1, 1)

        # Convert labels to one-hot if necessary (only for multi-class classification)
        if len(y_true.shape) == 1 and y_pred.ndim > 1 and y_pred.shape[1] > 1:
            n_outputs = y_pred.shape[1]
            y_true_ohe = np.zeros_like(y_pred)
            y_true_int = y_true.astype(int)
            y_true_ohe[range(sample_size), y_true_int] = 1
        else:
            y_true_ohe = y_true

        # derivative of mean(square(y_pred - y_true_ohe))
        # d/dy_pred = 2*(y_pred - y_true_ohe) / (number_elements)
        n_elements = np.prod(y_pred.shape)
        self.dinputs = 2 * (y_pred - y_true_ohe) / n_elements

        # Add regularization gradient to layer if provided
        if layer is not None:
            if not hasattr(layer, 'dweight'):
                layer.dweight = np.zeros_like(layer.weights)
            if not hasattr(layer, 'dbiases'):
                layer.dbiases = np.zeros_like(layer.biases)

            if self.regularization_l2 > 0:
                layer.dweight += 2 * self.regularization_l2 * layer.weights
                layer.dbiases += 2 * self.regularization_l2 * layer.biases
            if self.regularization_l1 > 0:
                layer.dweight += self.regularization_l1 * np.sign(layer.weights)
                layer.dbiases += self.regularization_l1 * np.sign(layer.biases)

        return self.dinputs

    def _calculate_regularization_loss(self, layer):
        regularization_loss = 0
        if layer is not None:
            if self.regularization_l2 > 0:
                regularization_loss += self.regularization_l2 * np.sum(layer.weights**2)
                regularization_loss += self.regularization_l2 * np.sum(layer.biases**2)

            if self.regularization_l1 > 0:
                regularization_loss += self.regularization_l1 * np.sum(np.abs(layer.weights))
                regularization_loss += self.regularization_l1 * np.sum(np.abs(layer.biases))
        return regularization_loss

class MeanAbsoluteError(BaseLoss):
    def __init__(self, regularization_l2=0.0, regularization_l1=0.0):
        self.regularization_l2 = regularization_l2
        self.regularization_l1 = regularization_l1

    def _calculate_regularization_loss(self, layer):
        regularization_loss = 0
        if layer is not None:
            if self.regularization_l2 > 0:
                regularization_loss += self.regularization_l2 * np.sum(layer.weights**2)
                regularization_loss += self.regularization_l2 * np.sum(layer.biases**2)

            if self.regularization_l1 > 0:
                regularization_loss += self.regularization_l1 * np.sum(np.abs(layer.weights))
                regularization_loss += self.regularization_l1 * np.sum(np.abs(layer.biases))
        return regularization_loss

    def forward(self, y_pred, y_true, layer=None):
        # Forward for MAE: support both integer labels and full targets
        y_pred = np.array(y_pred)
        sample_size = len(y_pred)

        # Convert integer labels to one-hot if necessary (only for multi-class classification)
        if len(y_true.shape) == 1 and y_pred.ndim > 1 and y_pred.shape[1] > 1:
            n_outputs = y_pred.shape[1]
            y_true_ohe = np.zeros_like(y_pred)
            y_true_int = y_true.astype(int)  # Pastikan y_true adalah integer
            y_true_ohe[range(sample_size), y_true_int] = 1
        else:
            y_true_ohe = y_true

        absolute_error = np.abs(y_pred - y_true_ohe)
        data_loss = np.mean(absolute_error)

        regularization_loss = self._calculate_regularization_loss(layer)
        return data_loss + regularization_loss

    def backward(self, y_pred, y_true, layer=None):
        # Backward for MAE
        y_pred = np.array(y_pred)
        sample_size = len(y_pred)

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Convert integer labels to one-hot if necessary (only for multi-class classification)
        if len(y_true.shape) == 1 and y_pred.ndim > 1 and y_pred.shape[1] > 1:
            n_outputs = y_pred.shape[1]
            y_true_ohe = np.zeros_like(y_pred)
            y_true_int = y_true.astype(int)
            y_true_ohe[range(sample_size), y_true_int] = 1
        else:
            y_true_ohe = y_true

        n_elements = np.prod(y_pred.shape)
        # derivative of absolute value: sign(y_pred - y_true)
        self.dinputs = np.sign(y_pred - y_true_ohe) / n_elements

        if layer is not None:
            if not hasattr(layer, 'dweight'):
                layer.dweight = np.zeros_like(layer.weights)
            if not hasattr(layer, 'dbiases'):
                layer.dbiases = np.zeros_like(layer.biases)

            if self.regularization_l2 > 0:
                layer.dweight += 2 * self.regularization_l2 * layer.weights
                layer.dbiases += 2 * self.regularization_l2 * layer.biases
            if self.regularization_l1 > 0:
                layer.dweight += self.regularization_l1 * np.sign(layer.weights)
                layer.dbiases += self.regularization_l1 * np.sign(layer.biases)

        return self.dinputs

class CategoricalCrossentropy(BaseLoss):
    def __init__(self, regularization_l2=0.0, regularization_l1=0.0):
        self.regularization_l2 = regularization_l2
        self.regularization_l1 = regularization_l1

    def _calculate_regularization_loss(self, layer):
        regularization_loss = 0
        if layer is not None:
            if self.regularization_l2 > 0:
                regularization_loss += self.regularization_l2 * np.sum(layer.weights**2)
                regularization_loss += self.regularization_l2 * np.sum(layer.biases**2)

            if self.regularization_l1 > 0:
                regularization_loss += self.regularization_l1 * np.sum(np.abs(layer.weights))
                regularization_loss += self.regularization_l1 * np.sum(np.abs(layer.biases))
        return regularization_loss

    def forward(self, y_pred, y_true, layer=None):
        # Menghitung sample
        sample = len(y_pred)
        # Menghitung y_pred_clipped
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Menghitung correct_confidence berdasarkan y_pred_clipped dan y_true
        if len(y_true.shape) == 1:
            y_true_int = y_true.astype(int)
            correct_confidence = y_pred_clipped[range(sample), y_true_int]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        # Menghitung negative_log_likelihood
        negative_log_likelihood = -np.log(correct_confidence)
        # Menghitung data_loss dengan menggunakan np.mean
        data_loss = np.mean(negative_log_likelihood)
         
        # Menghitung regularization_loss berdasarkan layer
        regularization_loss = self._calculate_regularization_loss(layer)
        
        # Mengembalikan data_loss + regularization_loss
        return data_loss + regularization_loss

    def backward(self, y_pred, y_true, layer=None):
        # Menghitung sample_size
        sample_size = len(y_pred)
        # Menghitung y_pred_clipped
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Menghitung self.dinputs berdasarkan y_pred_clipped dan y_true
        if len(y_true.shape) == 1:
            y_true_int = y_true.astype(int)
            self.dinputs = np.zeros_like(y_pred)
            self.dinputs[range(sample_size), y_true_int] = -1 / y_pred_clipped[range(sample_size), y_true_int]
            self.dinputs = self.dinputs / sample_size 
        elif len(y_true.shape) == 2:
            self.dinputs = -y_true / y_pred_clipped
            self.dinputs = self.dinputs / sample_size 

        # Mengatur dweight dan dbiases pada layer
        if layer is not None:
            if not hasattr(layer, 'dweight'):
                layer.dweight = np.zeros_like(layer.weights)
            if not hasattr(layer, 'dbiases'):
                layer.dbiases = np.zeros_like(layer.biases)
                
            # Menghitung regularization_loss berdasarkan layer
            regularization_loss = self._calculate_regularization_loss(layer)
            
            # Menambahkan regularization_loss ke dweight dan dbiases
            if self.regularization_l2 > 0:
                layer.dweight += 2 * self.regularization_l2 * layer.weights
                layer.dbiases += 2 * self.regularization_l2 * layer.biases
            
            if self.regularization_l1 > 0:
                layer.dweight += self.regularization_l1 * np.sign(layer.weights)
                layer.dbiases += self.regularization_l1 * np.sign(layer.biases)

        return self.dinputs