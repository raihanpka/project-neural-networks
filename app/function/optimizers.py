import numpy as np

class Optimizer_StochasticGradientDescent:
    # Stochastic Gradient Descent
    def __init__(self, learning_rate=0.01, momentum=0.9): 
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocity_weights = None
        self.velocity_biases = None

    def update_params(self, layer):
        if self.velocity_weights is None:
            self.velocity_weights = np.zeros_like(layer.weights)
            self.velocity_biases = np.zeros_like(layer.biases)

        # Menggunakan momentum, penggunaan gradient descent akan berubah.
        # Jika kita tidak menolak gradient, maka akan mengarah pada arah gradient tersebut,
        # yang mana akan meningkatkan loss. Oleh karena itu, kita perlu mengubah arah gradient
        # menjadi arah yang diametral berlawanan agar loss menjadi paling rendah.
        self.velocity_weights = (
            self.momentum * self.velocity_weights - 
            self.learning_rate * layer.dweight
        )

        self.velocity_biases = (
            self.momentum * self.velocity_biases -
            self.learning_rate * layer.dbiases
        )
        
        layer.weights += self.velocity_weights
        layer.biases += self.velocity_biases

class Optimizer_Adam:
    # Adaptive Moment Estimation
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
        self.epsilon = epsilon 
        self.m_weights = None 
        self.v_weights = None 
        self.m_biases = None  
        self.v_biases = None  
        self.t = 0 

    def update_params(self, layer):
        if self.m_weights is None:
            self.m_weights = np.zeros_like(layer.weights)
            self.v_weights = np.zeros_like(layer.weights)
            self.m_biases = np.zeros_like(layer.biases)
            self.v_biases = np.zeros_like(layer.biases)
        
        self.t += 1 # Increment dari waktu
        
        # Update m (mean) dan v (variance)
        self.m_weights = self.beta_1 * self.m_weights + (1 - self.beta_1) * layer.dweight
        self.v_weights = self.beta_2 * self.v_weights + (1 - self.beta_2) * np.square(layer.dweight)
        
        # Correct bias m (mean) dan v (variance)
        m_weights_corrected = self.m_weights / (1 - self.beta_1 ** self.t)
        v_weights_corrected = self.v_weights / (1 - self.beta_2 ** self.t)
        
        # Update weights
        layer.weights -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
        
        self.m_biases = self.beta_1 * self.m_biases + (1 - self.beta_1) * layer.dbiases
        self.v_biases = self.beta_2 * self.v_biases + (1 - self.beta_2) * np.square(layer.dbiases)
        
        # Correct bias m (mean) dan v (variance)
        m_biases_corrected = self.m_biases / (1 - self.beta_1 ** self.t)
        v_biases_corrected = self.v_biases / (1 - self.beta_2 ** self.t)
        
        # Update biases
        layer.biases -= self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon)