import numpy as np

class BaseRegularization:
    def forward(self, inputs, training=True):
        """
        Melakukan proses forward untuk regularization.
        """
        raise NotImplementedError
    
    def backward(self, dvalues):
        """
        Melakukan proses backward untuk regularization.
        """
        raise NotImplementedError

class BatchNormalization(BaseRegularization):
    def __init__(self, epsilon=1e-5, momentum=0.9, learning_rate=0.01):
        """
        Inisialisasi parameter BatchNormalization.
        """
        self.epsilon = epsilon  # Parameter untuk menghindari dividenya nol
        self.momentum = momentum  # Parameter untuk mengatur momentum
        self.learning_rate = learning_rate  # Learning rate untuk gradient descent
        self.running_mean = None  # Mean pada training
        self.running_var = None   # Var pada training
        self.gamma = None        # Scale atau shift
        self.beta = None         # Scale atau shift
        self.dgamma = None
        self.dbeta = None

    def forward(self, inputs, training=True):
        """
        Melakukan proses forward untuk BatchNormalization.
        """
        self.inputs = inputs
        input_shape = inputs.shape
        
        if self.gamma is None:
            self.gamma = np.ones(input_shape[1])  # Inisialisasi scale dan shift
            self.beta = np.zeros(input_shape[1])
            self.running_mean = np.zeros(input_shape[1])
            self.running_var = np.ones(input_shape[1])
        
        if training:
            mean = np.mean(inputs, axis=0)  # Menghitung mean
            var = np.var(inputs, axis=0)  # Menghitung var
            
            if self.running_mean is None:
                self.running_mean = mean
                self.running_var = var
            else:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            self.x_centered = inputs - mean  # Menghitung x_centered
            self.std = np.sqrt(var + self.epsilon)  # Menghitung std
            self.x_norm = self.x_centered / self.std  # Menghitung x_norm
        else:
            self.x_norm = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        self.output = self.gamma * self.x_norm + self.beta
        
        return self.output

    def backward(self, dvalues):
        """
        Melakukan proses backward untuk BatchNormalization dengan Gradient Descent.
        """
        batch_size = dvalues.shape[0]
        
        # Hitung gradient (Backpropagation)
        self.dgamma = np.sum(dvalues * self.x_norm, axis=0)  # Menghitung dgamma
        self.dbeta = np.sum(dvalues, axis=0)  # Menghitung dbeta
        
        # Gradient Clipping untuk stabilitas
        self.dgamma = np.clip(self.dgamma, -1.0, 1.0)
        self.dbeta = np.clip(self.dbeta, -1.0, 1.0)
        
        # GRADIENT DESCENT untuk gamma dan beta
        # γ = γ - α × ∇L/∂γ
        # β = β - α × ∇L/∂β
        self.gamma -= self.learning_rate * self.dgamma
        self.beta -= self.learning_rate * self.dbeta
        
        dx_norm = dvalues * self.gamma  # Menghitung dx_norm
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * self.std**(-3), axis=0)  # Menghitung dvar
        dmean = np.sum(dx_norm * -1 / self.std, axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)  # Menghitung dmean
        self.dinputs = dx_norm / self.std + dvar * 2 * self.x_centered / batch_size + dmean / batch_size  # Menghitung dinputs
        
        return self.dinputs

class Dropout(BaseRegularization):
    def __init__(self, rate):
        """
        Inisialisasi parameter Dropout.
        """
        self.rate = rate  # Probabilitas dropout
        self.binary_mask = None  # Mask binary
        
    def forward(self, inputs, training=True):
        """
        Melakukan proses forward untuk Dropout.
        """
        self.inputs = inputs
        
        if not training:
            self.output = inputs  # Jika training false, output = inputs
            return self.output
            
        self.binary_mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)  # Menghasilkan mask binary
        
        self.output = inputs * self.binary_mask  # Menghasilkan output dengan mask binary
        return self.output
        
    def backward(self, dvalues):
        """
        Melakukan proses backward untuk Dropout.
        """
        self.dinputs = dvalues * self.binary_mask  # Menghitung dinputs
        return self.dinputs