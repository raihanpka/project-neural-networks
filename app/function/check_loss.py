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
    def forward(self, y_pred, y_true, layer=None):
        # Menghitung sample
        sample = len(y_pred)
        # Menghitung y_pred_clipped
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Menghitung correct_confidence berdasarkan y_pred_clipped dan y_true
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        # Menghitung squared_error
        squared_error = np.square(correct_confidence - y_true)
        # Menghitung data_loss dengan menggunakan np.mean
        data_loss = np.mean(squared_error)
         
        # Menghitung regularization_loss berdasarkan layer
        regularization_loss = self._calculate_regularization_loss(layer)
        
        # Mengembalikan data_loss + regularization_loss
        return data_loss + regularization_loss

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
    def forward(self, y_pred, y_true, layer=None):
        # Menghitung sample
        sample = len(y_pred)
        # Menghitung y_pred_clipped
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Menghitung correct_confidence berdasarkan y_pred_clipped dan y_true
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        # Menghitung absolute_error
        absolute_error = np.abs(correct_confidence - y_true)
        # Menghitung data_loss dengan menggunakan np.mean
        data_loss = np.mean(absolute_error)
         
        # Menghitung regularization_loss berdasarkan layer
        regularization_loss = self._calculate_regularization_loss(layer)
        
        # Mengembalikan data_loss + regularization_loss
        return data_loss + regularization_loss

class CategoricalCrossentropy(BaseLoss):
    def __init__(self, regularization_l2=0.0, regularization_l1=0.0):
        self.regularization_l2 = regularization_l2
        self.regularization_l1 = regularization_l1

    def forward(self, y_pred, y_true, layer=None):
        # Menghitung sample
        sample = len(y_pred)
        # Menghitung y_pred_clipped
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Menghitung correct_confidence berdasarkan y_pred_clipped dan y_true
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
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
            self.dinputs = np.zeros_like(y_pred)
            self.dinputs[range(sample_size), y_true] = -1 / y_pred_clipped[range(sample_size), y_true]
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