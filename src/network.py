import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mnist import MNIST
from typing import Tuple, Dict, List
import copy
import os

# Set plotting style
sns.set_style('whitegrid')

class NeuralNetwork:
    def __init__(
        self,
        input_size: int = 784,      # 28x28 pixels
        hidden_size: int = 128,     # Number of neurons in hidden layer
        output_size: int = 10,      # Number of classes (digits 0-9)
        learning_rate: float = 0.01
    ):
        # Initialize network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        np.random.seed(42)  # For reproducibility
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        # Initialize history for visualization
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = [] 

        # Update visualization tracking with separate frequency maxima
        self.viz_stats = {
            'activations': {
                'hidden_min': float('inf'),
                'hidden_max': float('-inf'),
                'hidden_freq_max': 0,  # Separate frequency max for hidden layer
                'output_min': float('inf'),
                'output_max': float('-inf'),
                'output_freq_max': 0   # Separate frequency max for output layer
            },
            'gradients': {
                'layer1_min': float('inf'),
                'layer1_max': float('-inf'),
                'layer1_freq_max': 0,  # Separate frequency max for layer 1
                'layer2_min': float('inf'),
                'layer2_max': float('-inf'),
                'layer2_freq_max': 0    # Separate frequency max for layer 2
            },
            'confidence': {
                'correct_min': float('inf'),
                'correct_max': float('-inf'),
                'correct_freq_max': 0,   # Separate frequency max for correct predictions
                'incorrect_min': float('inf'),
                'incorrect_max': float('-inf'),
                'incorrect_freq_max': 0   # Separate frequency max for incorrect predictions
            }
        }
        
        # Store visualizations for animation
        self.viz_frames = {
            'activations': [],
            'gradients': [],
            'confidence': []
        }

    @staticmethod
    def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the MNIST dataset
        Returns: (X_train, y_train, X_test, y_test)
        """
        try:
            mndata = MNIST('./data')
            X_train, y_train = mndata.load_training()
            X_test, y_test = mndata.load_testing()
        except FileNotFoundError:
            raise FileNotFoundError(
                "MNIST dataset not found. Please run setup.sh first to download the dataset."
            )
        except Exception as e:
            raise Exception(f"Error loading MNIST dataset: {str(e)}")

        # Convert to numpy arrays and normalize
        X_train = np.array(X_train).astype(float) / 255.0
        X_test = np.array(X_test).astype(float) / 255.0
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        return X_train, y_train, X_test, y_test

    def add_explanation_to_figure(self, fig: plt.Figure, grid: plt.GridSpec, title: str, explanation: str) -> None:
        """
        Adds a text box with explanation to the figure using GridSpec
        """
        # Use the bottom row of the GridSpec for text
        text_ax = fig.add_subplot(grid[-1, :])
        text_ax.axis('off')
        
        # Add title and explanation
        text_ax.text(0.5, 1.0, title, 
                    fontsize=12, fontweight='bold', 
                    ha='center', va='top')
        
        text_ax.text(0.02, 0.7, explanation, 
                    fontsize=10, linespacing=1.5,
                    ha='left', va='top',
                    wrap=True)

    def visualize_sample_images(self, X: np.ndarray, y: np.ndarray, num_samples: int = 9) -> None:
        """Display a grid of sample images with their labels"""
        os.makedirs('visualizations/samples', exist_ok=True)
        
        fig = plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            ax = fig.add_subplot(3, 3, i + 1)
            ax.imshow(X[i].reshape(28, 28), cmap='gray')
            ax.set_title(f'Label: {y[i]}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/samples/mnist_samples.png')
        plt.close()

    def visualize_weight_distribution(self) -> None:
        """Plot histograms of weight distributions"""
        os.makedirs('visualizations/weights', exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.histplot(data=self.W1.flatten(), bins=50, ax=ax1)
        ax1.set_title('Layer 1 Weights')
        
        sns.histplot(data=self.W2.flatten(), bins=50, ax=ax2)
        ax2.set_title('Layer 2 Weights')
        
        plt.tight_layout()
        plt.savefig('visualizations/weights/distribution.png')
        plt.close()

    def relu(self, Z: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, Z)

    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation function"""
        return Z > 0

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward_propagation(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward propagation step
        Returns: (predictions, cache)
        """
        # First layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        
        # Second layer
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        
        # Store values for backpropagation
        cache = {
            'X': X,
            'Z1': Z1, 'A1': A1,
            'Z2': Z2, 'A2': A2
        }
        
        return A2, cache

    def compute_loss(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """
        Compute cross-entropy loss
        """
        m = Y_true.shape[0]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
        return loss

    def backward_propagation(self, cache: Dict, Y_true: np.ndarray) -> Dict:
        """
        Backward propagation step
        Returns: gradients dictionary
        """
        m = Y_true.shape[0]
        
        # Output layer gradients
        dZ2 = cache['A2'] - Y_true
        dW2 = np.dot(cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(cache['Z1'])
        dW1 = np.dot(cache['X'].T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def update_parameters(self, gradients: Dict) -> None:
        """
        Update network parameters using gradient descent
        """
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']

    def update_viz_stats(self, data_type: str, current_data: Dict) -> None:
        """Update visualization statistics for consistent scaling"""
        stats = self.viz_stats[data_type]
        
        if data_type == 'activations':
            hidden_data = current_data['A1'].flatten()
            output_data = current_data['A2'].flatten()
            
            stats['hidden_min'] = min(stats['hidden_min'], np.min(hidden_data))
            stats['hidden_max'] = max(stats['hidden_max'], np.max(hidden_data))
            stats['output_min'] = min(stats['output_min'], np.min(output_data))
            stats['output_max'] = max(stats['output_max'], np.max(output_data))
            
            # Update frequency maxima separately
            hist_hidden, _ = np.histogram(hidden_data, bins=50)
            hist_output, _ = np.histogram(output_data, bins=50)
            stats['hidden_freq_max'] = max(stats['hidden_freq_max'], np.max(hist_hidden))
            stats['output_freq_max'] = max(stats['output_freq_max'], np.max(hist_output))
            
        elif data_type == 'gradients':
            grad1_data = np.abs(current_data['dW1']).flatten()
            grad2_data = np.abs(current_data['dW2']).flatten()
            
            stats['layer1_min'] = min(stats['layer1_min'], np.min(grad1_data))
            stats['layer1_max'] = max(stats['layer1_max'], np.max(grad1_data))
            stats['layer2_min'] = min(stats['layer2_min'], np.min(grad2_data))
            stats['layer2_max'] = max(stats['layer2_max'], np.max(grad2_data))
            
            # Update frequency maxima separately
            hist_grad1, _ = np.histogram(grad1_data, bins=50)
            hist_grad2, _ = np.histogram(grad2_data, bins=50)
            stats['layer1_freq_max'] = max(stats['layer1_freq_max'], np.max(hist_grad1))
            stats['layer2_freq_max'] = max(stats['layer2_freq_max'], np.max(hist_grad2))
            
        elif data_type == 'confidence':
            correct_data = current_data['correct']
            incorrect_data = current_data.get('incorrect', np.array([]))
            
            # Update min/max for correct predictions
            stats['correct_min'] = min(stats['correct_min'], np.min(correct_data))
            stats['correct_max'] = max(stats['correct_max'], np.max(correct_data))
            
            # Update frequency max for correct predictions
            hist_correct, _ = np.histogram(correct_data, bins=50)
            stats['correct_freq_max'] = max(stats['correct_freq_max'], np.max(hist_correct))
            
            # Update stats for incorrect predictions if they exist
            if len(incorrect_data) > 0:
                stats['incorrect_min'] = min(stats['incorrect_min'], np.min(incorrect_data))
                stats['incorrect_max'] = max(stats['incorrect_max'], np.max(incorrect_data))
                hist_incorrect, _ = np.histogram(incorrect_data, bins=50)
                stats['incorrect_freq_max'] = max(stats['incorrect_freq_max'], np.max(hist_incorrect))
        
        elif data_type == 'training':
            stats['loss_min'] = min(stats['loss_min'], min(self.loss_history))
            stats['loss_max'] = max(stats['loss_max'], max(self.loss_history))
            stats['acc_min'] = min(stats['acc_min'], min(self.accuracy_history))
            stats['acc_max'] = max(stats['acc_max'], max(self.accuracy_history))

    def visualize_layer_activations(self, cache: Dict, epoch: int) -> None:
        """Visualize layer activations with histograms"""
        epoch_dir = f'visualizations/epoch_{epoch:03d}'
        os.makedirs(epoch_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Hidden layer activations
        sns.histplot(data=cache['A1'].flatten(), bins=50, ax=ax1)
        ax1.set_title('Hidden Layer Activations')
        ax1.set_xlabel('Activation Value')
        ax1.set_ylabel('Frequency')
        
        # Output layer activations
        sns.histplot(data=cache['A2'].flatten(), bins=50, ax=ax2)
        ax2.set_title('Output Layer Activations')
        ax2.set_xlabel('Activation Value')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{epoch_dir}/activations.png')
        plt.close()

    def visualize_gradients(self, gradients: Dict, epoch: int) -> None:
        """Visualize gradient magnitudes with histograms"""
        epoch_dir = f'visualizations/epoch_{epoch:03d}'
        os.makedirs(epoch_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Layer 1 gradients
        sns.histplot(data=np.abs(gradients['dW1']).flatten(), bins=50, ax=ax1)
        ax1.set_title('Layer 1 Gradient Magnitudes')
        ax1.set_xlabel('Gradient Magnitude')
        ax1.set_ylabel('Frequency')
        
        # Layer 2 gradients
        sns.histplot(data=np.abs(gradients['dW2']).flatten(), bins=50, ax=ax2)
        ax2.set_title('Layer 2 Gradient Magnitudes')
        ax2.set_xlabel('Gradient Magnitude')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{epoch_dir}/gradients.png')
        plt.close()

    def visualize_training_progress(self, epoch: int) -> None:
        """Visualize training metrics (loss and accuracy)"""
        epoch_dir = f'visualizations/epoch_{epoch:03d}'
        os.makedirs(epoch_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        # Accuracy plot
        ax2.plot(self.accuracy_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'{epoch_dir}/training_progress.png')
        plt.close()

    def save_training_snapshot(self, epoch: int, cache: Dict, 
                             gradients: Dict, predictions: np.ndarray, 
                             true_labels: np.ndarray) -> None:
        """Save all visualizations for current epoch"""
        self.visualize_training_progress(epoch)
        self.visualize_layer_activations(cache, epoch)
        self.visualize_gradients(gradients, epoch)

    def create_animation(self, viz_type: str) -> None:
        """Create animation from stored frames"""
        if not self.viz_frames[viz_type]:
            print(f"No frames available for {viz_type}")
            return
        
        import matplotlib.animation as animation
        
        # Create a new figure for the animation
        fig = plt.figure(figsize=(15, 10))
        
        def animate(frame):
            # Clear the current figure
            plt.clf()
            
            # Copy content from the saved frame
            for ax_orig in self.viz_frames[viz_type][frame].axes:
                # Create new axis with same position
                new_ax = fig.add_axes(ax_orig.get_position())
                
                if ax_orig.has_data():
                    # For histograms
                    if len(ax_orig.patches) > 0:
                        # Get the bin edges and heights from the original histogram
                        patches = ax_orig.patches
                        bin_edges = np.array([p.get_x() for p in patches] + 
                                           [patches[-1].get_x() + patches[-1].get_width()])
                        bin_heights = [p.get_height() for p in patches]
                        
                        # Recreate the histogram
                        new_ax.hist(bin_edges[:-1], bins=bin_edges, weights=bin_heights,
                                  density=False, alpha=0.7)
                    
                    # For line plots
                    for line in ax_orig.lines:
                        new_ax.plot(line.get_xdata(), line.get_ydata(),
                                  color=line.get_color(),
                                  linestyle=line.get_linestyle())
                    
                    # Copy axis properties
                    new_ax.set_xlim(ax_orig.get_xlim())
                    new_ax.set_ylim(ax_orig.get_ylim())
                    new_ax.set_title(ax_orig.get_title())
                    new_ax.set_xlabel(ax_orig.get_xlabel())
                    new_ax.set_ylabel(ax_orig.get_ylabel())
                else:
                    # For text-only axes (explanations)
                    new_ax.axis('off')
                    for text in ax_orig.texts:
                        new_ax.text(text.get_position()[0], text.get_position()[1],
                                  text.get_text(),
                                  fontsize=text.get_fontsize(),
                                  ha=text.get_ha(), va=text.get_va())
            
            return fig.axes
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(self.viz_frames[viz_type]),
            interval=500,  # 500ms between frames
            blit=True
        )
        
        # Save animation
        anim.save(f'visualizations/{viz_type}_animation.gif',
                 writer='pillow', fps=2)
        plt.close()

    def visualize_gradient_flow(self, gradients: Dict, epoch: int) -> None:
        """Visualize gradient magnitudes with consistent scaling"""
        # Update statistics
        self.update_viz_stats('gradients', gradients)
        stats = self.viz_stats['gradients']
        
        fig = plt.figure(figsize=(15, 10))
        grid = plt.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.3)
        
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        
        # Calculate statistics for explanation
        grad1_mean = np.mean(np.abs(gradients['dW1']))
        grad2_mean = np.mean(np.abs(gradients['dW2']))
        
        # Layer 1 gradients
        sns.histplot(data=np.abs(gradients['dW1']).flatten(), bins=50, kde=True, ax=ax1)
        ax1.set_title('Layer 1 Gradient Magnitudes')
        ax1.set_xlabel('Gradient Magnitude')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(stats['layer1_min'], stats['layer1_max'])
        ax1.set_ylim(0, stats['layer1_freq_max'])
        
        # Layer 2 gradients
        sns.histplot(data=np.abs(gradients['dW2']).flatten(), bins=50, kde=True, ax=ax2)
        ax2.set_title('Layer 2 Gradient Magnitudes')
        ax2.set_xlabel('Gradient Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim(stats['layer2_min'], stats['layer2_max'])
        ax2.set_ylim(0, stats['layer2_freq_max'])
        
        explanation = f"""
        Gradient Flow Analysis (Epoch {epoch})
        
        Layer 1:
        • Mean gradient magnitude: {grad1_mean:.6f}
        • Range: [{stats['layer1_min']:.6f}, {stats['layer1_max']:.6f}]
        
        Layer 2:
        • Mean gradient magnitude: {grad2_mean:.6f}
        • Range: [{stats['layer2_min']:.6f}, {stats['layer2_max']:.6f}]
        
        {'Warning: Potential vanishing gradients' if grad1_mean < 1e-6 else 
         'Warning: Potential exploding gradients' if grad1_mean > 1e2 else 
         'Gradient magnitudes appear healthy'}
        """
        
        self.add_explanation_to_figure(fig, grid, "Gradient Flow", explanation)
        
        # Save for animation
        self.viz_frames['gradients'].append(fig)
        
        # Save individual frame
        plt.savefig(f'visualizations/gradients_epoch_{epoch:03d}.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_prediction_confidence(self, predictions: np.ndarray, true_labels: np.ndarray, epoch: int) -> None:
        """Visualize prediction confidence with consistent scaling"""
        # Get confidence data
        confidence = np.max(predictions, axis=1)
        predicted_classes = np.argmax(predictions, axis=1)
        correct_predictions = predicted_classes == true_labels
        
        # Prepare data for stats update
        confidence_data = {
            'correct': confidence[correct_predictions],
            'incorrect': confidence[~correct_predictions] if not np.all(correct_predictions) else np.array([])
        }
        
        # Update statistics
        self.update_viz_stats('confidence', confidence_data)
        stats = self.viz_stats['confidence']
        
        fig = plt.figure(figsize=(15, 10))
        grid = plt.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.3)
        
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        
        # Calculate statistics for explanation
        correct_mean = np.mean(confidence_data['correct'])
        incorrect_mean = np.mean(confidence_data['incorrect']) if len(confidence_data['incorrect']) > 0 else 0
        
        # Correct predictions
        sns.histplot(data=confidence_data['correct'], bins=50, kde=True, ax=ax1, color='g')
        ax1.set_title('Confidence Distribution (Correct Predictions)')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(stats['correct_min'], stats['correct_max'])
        ax1.set_ylim(0, stats['correct_freq_max'])
        
        # Incorrect predictions
        if len(confidence_data['incorrect']) > 0:
            sns.histplot(data=confidence_data['incorrect'], bins=50, kde=True, ax=ax2, color='r')
            ax2.set_title('Confidence Distribution (Incorrect Predictions)')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Frequency')
            ax2.set_xlim(stats['incorrect_min'], stats['incorrect_max'])
            ax2.set_ylim(0, stats['incorrect_freq_max'])
        
        explanation = f"""
        Prediction Confidence Analysis (Epoch {epoch})
        
        Correct Predictions:
        • Mean confidence: {correct_mean:.3f}
        • Range: [{stats['correct_min']:.3f}, {stats['correct_max']:.3f}]
        
        Incorrect Predictions:
        • Mean confidence: {f'{incorrect_mean:.3f}' if len(confidence_data['incorrect']) > 0 else 'N/A'}
        • Count: {len(confidence_data['incorrect'])} predictions
        
        {'Warning: High confidence in wrong predictions' if incorrect_mean > 0.8 else
         'Warning: Low confidence overall' if correct_mean < 0.5 else
         'Confidence distribution appears healthy'}
        """
        
        self.add_explanation_to_figure(fig, grid, "Prediction Confidence", explanation)
        
        # Save for animation
        self.viz_frames['confidence'].append(fig)
        
        # Save individual frame
        plt.savefig(f'visualizations/confidence_epoch_{epoch:03d}.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()

    def one_hot_encode(self, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """Convert integer labels to one-hot encoded format"""
        return np.eye(num_classes)[y]

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 10, batch_size: int = 32, 
              viz_interval: int = 1) -> None:
        """
        Train the neural network and save visualizations at specified intervals
        """
        num_samples = X_train.shape[0]
        num_batches = num_samples // batch_size
        y_train_onehot = self.one_hot_encode(y_train)
        
        # Store visualization data for all epochs
        viz_data = {
            'epochs': [],
            'activations': {'hidden': [], 'output': []},
            'gradients': {'layer1': [], 'layer2': []},
            'metrics': {'loss': [], 'accuracy': []}
        }
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct_predictions = 0
            
            # Training loop
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                predictions, cache = self.forward_propagation(X_batch)
                loss = self.compute_loss(predictions, y_batch)
                gradients = self.backward_propagation(cache, y_batch)
                self.update_parameters(gradients)
                
                epoch_loss += loss
                correct_predictions += np.sum(
                    np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1)
                )
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / num_batches
            accuracy = correct_predictions / (num_batches * batch_size)
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Store data for visualization at specified intervals
            if epoch % viz_interval == 0:
                viz_data['epochs'].append(epoch)
                viz_data['activations']['hidden'].append(cache['A1'].flatten())
                viz_data['activations']['output'].append(cache['A2'].flatten())
                viz_data['gradients']['layer1'].append(np.abs(gradients['dW1']).flatten())
                viz_data['gradients']['layer2'].append(np.abs(gradients['dW2']).flatten())
                viz_data['metrics']['loss'].append(avg_loss)
                viz_data['metrics']['accuracy'].append(accuracy)
        
        # Calculate global min/max values for scaling
        scale_limits = self._calculate_scale_limits(viz_data)
        
        # Generate visualizations with consistent scaling
        self._generate_visualizations(viz_data, scale_limits)

    def _calculate_scale_limits(self, viz_data: Dict) -> Dict:
        """Calculate global min/max values for consistent scaling"""
        limits = {
            'activations': {
                'hidden': {'max_density': 0, 'max_val': 0, 'min_val': float('inf')},
                'output': {'max_density': 0, 'max_val': 0, 'min_val': float('inf')}
            },
            'gradients': {
                'layer1': {'max_density': 0, 'max_val': 0, 'min_val': float('inf')},
                'layer2': {'max_density': 0, 'max_val': 0, 'min_val': float('inf')}
            }
        }
        
        # Calculate KDE for each epoch to find global max density
        for epoch_idx in range(len(viz_data['epochs'])):
            # Activations
            for layer in ['hidden', 'output']:
                data = viz_data['activations'][layer][epoch_idx]
                kde = sns.kdeplot(data=data, bw_adjust=0.5).get_lines()[0].get_data()
                limits['activations'][layer]['max_density'] = max(
                    limits['activations'][layer]['max_density'],
                    np.max(kde[1])
                )
                limits['activations'][layer]['max_val'] = max(
                    limits['activations'][layer]['max_val'],
                    np.max(data)
                )
                limits['activations'][layer]['min_val'] = min(
                    limits['activations'][layer]['min_val'],
                    np.min(data)
                )
                plt.close()  # Close the temporary plot
            
            # Gradients
            for layer in ['layer1', 'layer2']:
                data = viz_data['gradients'][layer][epoch_idx]
                kde = sns.kdeplot(data=data, bw_adjust=0.5).get_lines()[0].get_data()
                limits['gradients'][layer]['max_density'] = max(
                    limits['gradients'][layer]['max_density'],
                    np.max(kde[1])
                )
                limits['gradients'][layer]['max_val'] = max(
                    limits['gradients'][layer]['max_val'],
                    np.max(data)
                )
                limits['gradients'][layer]['min_val'] = min(
                    limits['gradients'][layer]['min_val'],
                    np.min(data)
                )
                plt.close()  # Close the temporary plot
        
        return limits

    def _generate_visualizations(self, viz_data: Dict, scale_limits: Dict) -> None:
        """Generate visualizations with consistent scaling"""
        for epoch_idx, epoch in enumerate(viz_data['epochs']):
            epoch_dir = f'visualizations/epoch_{epoch:03d}'
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Activations
            fig = plt.figure(figsize=(15, 10))
            grid = plt.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(grid[0, 0])
            ax2 = fig.add_subplot(grid[0, 1])
            
            # Get data
            hidden_data = viz_data['activations']['hidden'][epoch_idx]
            output_data = viz_data['activations']['output'][epoch_idx]
            
            # Plot density plots instead of histograms
            sns.kdeplot(
                data=hidden_data,
                ax=ax1,
                color='blue',
                fill=True,
                alpha=0.6,
                linewidth=2,
                bw_adjust=0.5  # Consistent bandwidth
            )
            ax1.set_title('Hidden Layer Activations', fontsize=12, pad=10)
            ax1.set_xlabel('Activation Value', fontsize=10)
            ax1.set_ylabel('Density', fontsize=10)
            ax1.set_xlim(scale_limits['activations']['hidden']['min_val'],
                        scale_limits['activations']['hidden']['max_val'])
            ax1.set_ylim(0, scale_limits['activations']['hidden']['max_density'])
            
            sns.kdeplot(
                data=output_data,
                ax=ax2,
                color='red',
                fill=True,
                alpha=0.6,
                linewidth=2,
                bw_adjust=0.5  # Consistent bandwidth
            )
            ax2.set_title('Output Layer Activations (Softmax)', fontsize=12, pad=10)
            ax2.set_xlabel('Activation Value', fontsize=10)
            ax2.set_ylabel('Density', fontsize=10)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, scale_limits['activations']['output']['max_density'])
            
            # Calculate statistics for analysis
            dead_neurons = np.mean(hidden_data == 0) * 100
            max_confidence = np.max(output_data)
            
            analysis = f"""Hidden Layer (Left):
• {dead_neurons:.1f}% neurons are "dead" (zero activation)
• Ideal: Bell-shaped with moderate values
• Warning: {
    'High number of dead neurons' if dead_neurons > 20
    else 'Healthy activation pattern'
}

Output Layer (Right):
• Max confidence: {max_confidence:.2f}
• Distribution shows {
    'very low confidence' if max_confidence < 0.3
    else 'moderate confidence' if max_confidence < 0.7
    else 'high confidence'
} predictions

Status: {
    'Healthy activation pattern' if dead_neurons < 20 and max_confidence > 0.8
    else 'Potential issues: High number of dead neurons' if dead_neurons >= 20
    else 'Potential issues: Low confidence predictions' if max_confidence <= 0.8
    else 'Potential optimization needed'
}"""
            
            # Add analysis text with improved formatting
            text_ax = fig.add_subplot(grid[1, :])
            text_ax.axis('off')
            text_ax.text(0.02, 0.95, analysis,
                        fontsize=11,
                        va='top',
                        ha='left',
                        linespacing=1.5,
                        bbox=dict(
                            facecolor='white',
                            alpha=0.8,
                            edgecolor='lightgray',
                            boxstyle='round,pad=1'
                        ))
            
            plt.tight_layout()
            plt.savefig(f'{epoch_dir}/activations.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Gradients visualization with KDE
            fig = plt.figure(figsize=(15, 10))
            grid = plt.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(grid[0, 0])
            ax2 = fig.add_subplot(grid[0, 1])
            
            grad1_data = viz_data['gradients']['layer1'][epoch_idx]
            grad2_data = viz_data['gradients']['layer2'][epoch_idx]
            
            sns.kdeplot(
                data=grad1_data,
                ax=ax1,
                color='purple',
                fill=True,
                alpha=0.6,
                linewidth=2,
                bw_adjust=0.5
            )
            ax1.set_title('Layer 1 Gradient Magnitudes', fontsize=12, pad=10)
            ax1.set_xlabel('Gradient Magnitude', fontsize=10)
            ax1.set_ylabel('Density', fontsize=10)
            ax1.set_xlim(scale_limits['gradients']['layer1']['min_val'],
                        scale_limits['gradients']['layer1']['max_val'])
            ax1.set_ylim(0, scale_limits['gradients']['layer1']['max_density'])
            
            sns.kdeplot(
                data=grad2_data,
                ax=ax2,
                color='green',
                fill=True,
                alpha=0.6,
                linewidth=2,
                bw_adjust=0.5
            )
            ax2.set_title('Layer 2 Gradient Magnitudes', fontsize=12, pad=10)
            ax2.set_xlabel('Gradient Magnitude', fontsize=10)
            ax2.set_ylabel('Density', fontsize=10)
            ax2.set_xlim(scale_limits['gradients']['layer2']['min_val'],
                        scale_limits['gradients']['layer2']['max_val'])
            ax2.set_ylim(0, scale_limits['gradients']['layer2']['max_density'])
            
            # Add gradient analysis...
            grad_analysis = f"""Layer 1 Gradients (Left):
• Mean magnitude: {np.mean(grad1_data):.6f}
• Max magnitude: {np.max(grad1_data):.6f}
• Distribution shows gradient flow strength

Layer 2 Gradients (Right):
• Mean magnitude: {np.mean(grad2_data):.6f}
• Max magnitude: {np.max(grad2_data):.6f}
• Should be similar scale to Layer 1

Status: {
    'Healthy gradient flow' if 1e-6 < np.mean(grad1_data) < 1 and 1e-6 < np.mean(grad2_data) < 1
    else 'Warning: Potential vanishing gradients' if np.mean(grad1_data) < 1e-6 or np.mean(grad2_data) < 1e-6
    else 'Warning: Potential exploding gradients' if np.mean(grad1_data) > 1 or np.mean(grad2_data) > 1
    else 'Unusual gradient pattern'
}"""
            
            text_ax = fig.add_subplot(grid[1, :])
            text_ax.axis('off')
            text_ax.text(0.02, 0.95, grad_analysis,
                        fontsize=11,
                        va='top',
                        ha='left',
                        linespacing=1.5,
                        bbox=dict(
                            facecolor='white',
                            alpha=0.8,
                            edgecolor='lightgray',
                            boxstyle='round,pad=1'
                        ))
            
            plt.tight_layout()
            plt.savefig(f'{epoch_dir}/gradients.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Training Progress
            fig = plt.figure(figsize=(15, 10))
            grid = plt.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(grid[0, 0])
            ax2 = fig.add_subplot(grid[0, 1])
            
            # Plot full history up to current epoch
            epochs_range = range(epoch_idx + 1)
            loss_history = viz_data['metrics']['loss'][:epoch_idx + 1]
            acc_history = viz_data['metrics']['accuracy'][:epoch_idx + 1]
            
            ax1.plot(epochs_range, loss_history, 'b-', linewidth=2)
            ax1.set_title('Training Loss', fontsize=12, pad=10)
            ax1.set_xlabel('Epoch', fontsize=10)
            ax1.set_ylabel('Loss', fontsize=10)
            ax1.grid(True)
            
            ax2.plot(epochs_range, acc_history, 'r-', linewidth=2)
            ax2.set_title('Training Accuracy', fontsize=12, pad=10)
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Accuracy', fontsize=10)
            ax2.set_ylim(0, 1)
            ax2.grid(True)
            
            # Calculate training statistics
            current_loss = loss_history[-1]
            current_acc = acc_history[-1]
            
            if len(loss_history) > 1:
                loss_change = current_loss - loss_history[-2]
                acc_change = current_acc - acc_history[-2]
                loss_trend = np.mean(np.diff(loss_history[-5:]))  # Average change over last 5 epochs
                acc_trend = np.mean(np.diff(acc_history[-5:]))    # Average change over last 5 epochs
            else:
                loss_change = 0
                acc_change = 0
                loss_trend = 0
                acc_trend = 0
            
            training_analysis = f"""Training Progress (Epoch {epoch}):
• Current Loss: {current_loss:.4f} ({loss_change:+.4f})
• Current Accuracy: {current_acc:.2%} ({acc_change:+.2%})
• Recent Trend (last 5 epochs):
  - Loss: {'Decreasing' if loss_trend < 0 else 'Increasing'} by {abs(loss_trend):.4f} per epoch
  - Accuracy: {'Improving' if acc_trend > 0 else 'Declining'} by {abs(acc_trend):.2%} per epoch

Status: {
    'Good progress' if loss_trend < 0 and acc_trend > 0
    else 'Potential plateau' if abs(loss_trend) < 0.001 and abs(acc_trend) < 0.001
    else 'Warning: Training instability' if loss_trend > 0
    else 'Mixed results - monitor closely'
}"""
            
            # Add analysis text
            text_ax = fig.add_subplot(grid[1, :])
            text_ax.axis('off')
            text_ax.text(0.02, 0.95, training_analysis,
                        fontsize=11,
                        va='top',
                        ha='left',
                        linespacing=1.5,
                        bbox=dict(
                            facecolor='white',
                            alpha=0.8,
                            edgecolor='lightgray',
                            boxstyle='round,pad=1'
                        ))
            
            plt.tight_layout()
            plt.savefig(f'{epoch_dir}/training_progress.png', bbox_inches='tight', dpi=300)
            plt.close()