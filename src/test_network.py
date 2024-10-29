import numpy as np
from network import NeuralNetwork

def test_data_loading_and_visualization():
    # Initialize network
    nn = NeuralNetwork()
    
    # Load data
    X_train, y_train, X_test, y_test = nn.load_data()
    
    # Basic assertions
    assert X_train.shape[1] == 784, "Expected 784 features in training data"
    assert X_train.max() <= 1.0, "Data should be normalized between 0 and 1"
    assert X_train.min() >= 0.0, "Data should be normalized between 0 and 1"
    
    # Visualize samples
    nn.visualize_sample_images(X_train, y_train)
    
    # Visualize weight distributions
    nn.visualize_weight_distribution()
    
    print("Data loading and visualization test completed successfully!")

def test_forward_propagation():
    nn = NeuralNetwork()
    X_train, y_train, _, _ = nn.load_data()
    
    # Test with a small batch
    batch_size = 32
    X_batch = X_train[:batch_size]
    
    # Forward pass
    predictions, cache = nn.forward_propagation(X_batch)
    
    # Basic assertions
    assert predictions.shape == (batch_size, 10), "Wrong output shape"
    assert np.allclose(np.sum(predictions, axis=1), 1), "Softmax outputs should sum to 1"
    
    # Visualize activations
    nn.visualize_layer_activations(cache, epoch=0)
    
    print("Forward propagation test completed successfully!")

def test_training():
    nn = NeuralNetwork()
    X_train, y_train, _, _ = nn.load_data()
    
    # Test training with a small subset of data
    epochs = 100  # Match the epochs we're actually using
    nn.train(X_train[:1000], y_train[:1000],
             epochs=epochs, 
             batch_size=32,
             viz_interval=1)
    
    # Basic assertions
    assert len(nn.loss_history) == epochs, f"Should have {epochs} loss values"
    assert len(nn.accuracy_history) == epochs, f"Should have {epochs} accuracy values"
    assert all(0 <= acc <= 1 for acc in nn.accuracy_history), "Accuracy should be between 0 and 1"
    
    print("Training test completed successfully!")

if __name__ == "__main__":
    test_data_loading_and_visualization()
    test_forward_propagation()
    test_training()
