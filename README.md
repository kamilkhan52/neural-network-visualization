# Neural Network Training Visualization Tool

This project provides a comprehensive visualization suite for monitoring neural network training progress, specifically designed for a simple feedforward neural network trained on the MNIST dataset.

## Key Features
- Real-time visualization of network behavior during training
- PDF report generation with sequential visualizations
- Detailed analysis of activation patterns, gradient flow, and prediction confidence
- Automated MNIST dataset download and setup

## Components

### Network Architecture (`src/network.py`)
- 2-layer feedforward neural network
  - Input layer (784 neurons - 28x28 MNIST images)
  - Hidden layer (128 neurons with ReLU activation)
  - Output layer (10 neurons with Softmax activation)
- Cross-entropy loss function
- Mini-batch gradient descent

### Visualization Types

#### a) Activation Distributions
- Hidden layer ReLU activations
- Output layer Softmax distributions
- Dead neuron analysis
- Confidence tracking

#### b) Gradient Flow
- Layer-wise gradient magnitude distributions
- Vanishing/exploding gradient detection
- Training stability analysis

#### c) Training Progress
- Loss curve tracking
- Accuracy progression
- Per-epoch performance metrics

### PDF Report Generation (`create_training_slides.py`)
- Sequential visualization of training progression
- Maintains aspect ratios of plots
- Organized by visualization type
- Automatic file collection and sorting

## Setup and Usage

### Installation
```bash
pip install -r requirements.txt
./setup.sh
```

### Generate Training Visualizations
```bash
python src/test_network.py
```

### Create PDF Report
```bash
python create_training_slides.py
```

## Dependencies
- **numpy**: Neural network operations
- **matplotlib**: Plot generation
- **seaborn**: Statistical visualizations
- **reportlab**: PDF generation
- **Pillow**: Image processing
- **python-mnist**: MNIST dataset handling

## Project Structure
```
/
├── src/
│   ├── network.py          # Neural network implementation
│   └── test_network.py     # Test suite
├── visualizations/
│   ├── epoch_/            # Per-epoch visualization outputs
│   └── README.md          # Visualization interpretation guide
├── data/                  # MNIST dataset storage
├── create_training_slides.py  # PDF report generator
├── setup.sh              # Dataset download script
└── requirements.txt      # Project dependencies
```

## Output Structure

Each training epoch generates three key visualizations:
1. `activations.png`: Neural activation distributions
2. `gradients.png`: Gradient flow analysis
3. `training_progress.png`: Loss and accuracy curves

The final PDF report sequences these visualizations to show the progression of training over time.

## Visualization Interpretation

Refer to `visualizations/README.md` for detailed guidance on:
- What each visualization represents
- Key patterns to look for
- Warning signs of training issues
- Optimization opportunities