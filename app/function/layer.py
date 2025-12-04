import numpy as np

class BaseLayer:
    """Class ini adalah class dasar untuk semua layer"""
    def forward(self, inputs, training=True):
        raise NotImplementedError
    
    def backward(self, dvalues):
        raise NotImplementedError

class Dense(BaseLayer):
    def __init__(self, n_inputs, n_neurons, learning_rate=0.01):
        """
        Inisialisasi layer dense (fully connected)
        
        Parameters:
        -----------
        n_inputs : int
            Jumlah input/neuron dari layer sebelumnya
        n_neurons : int
            Jumlah neuron di layer ini
        learning_rate : float, default=0.01
            Learning rate untuk update bobot
        """
        # Inisialisasi weight dengan He initialization
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.learning_rate = learning_rate

    def forward(self, inputs, training=True):
        """
        Forward pass
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input data dengan shape (batch_size, n_inputs)
            
        Returns:
        --------
        numpy.ndarray
            Output dari layer ini
        """
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        """
        Backward pass dengan Gradient Descent
        
        Parameters:
        -----------
        dvalues : numpy.ndarray
            Gradien dari layer selanjutnya
            
        Returns:
        --------
        numpy.ndarray
            Gradien untuk layer sebelumnya
        """
        # Gradien terhadap weight dan bias (Backpropagation)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient Clipping untuk stabilitas training
        self.dweights = np.clip(self.dweights, -1.0, 1.0)
        self.dbiases = np.clip(self.dbiases, -1.0, 1.0)
        
        # Gradien untuk layer sebelumnya
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        # Gradient Descent: θ = θ - α × ∇L/∂θ
        # Update weights dan biases menggunakan gradient descent
        self.weights -= self.learning_rate * self.dweights
        self.biases -= self.learning_rate * self.dbiases
        
        return self.dinputs