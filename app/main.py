import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.neuralnetwork import NeuralNetwork
from app.function.layer import Dense
from app.function.activations import ReLU, Softmax
from app.function.regularization import BatchNormalization, Dropout
from app.function.check_loss import CategoricalCrossentropy
from app.function.metrics import calculate_accuracy
from app.data.dataset import create_data

def main():
    X, Y = create_data(samples=100, classes=3)
    
    model = NeuralNetwork()
    
    model.add(Dense(2, 128, learning_rate=0.002, optimizer='adam'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    
    model.add(Dense(128, 64, learning_rate=0.002, optimizer='adam'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    
    model.add(Dense(64, 32, learning_rate=0.002, optimizer='adam'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    
    model.add(Dense(32, 3, learning_rate=0.002, optimizer='adam'))
    model.add(Softmax())
    
    model.set_loss(CategoricalCrossentropy(regularization_l2=0.0001))
    
    print("Training on spiral dataset")
    epochs = int(input("Enter the number of epochs (recommended 500+): "))
    batch_size = int(input("Enter the batch size (recommended 16-32, or 0 for full batch): "))
    
    model.train(X, Y, epochs=epochs, batch_size=batch_size)
    
    predictions = model.predict_proba(X)
    accuracy = calculate_accuracy(Y, predictions)
    print(f"\nFinal Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()