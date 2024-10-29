#!/bin/bash

# Create directories
mkdir -p src
mkdir -p visualizations
mkdir -p data

# Download MNIST dataset if not already present
cd data

# Download training set
if [ ! -f "train-images-idx3-ubyte.gz" ]; then
    echo "Downloading MNIST training images..."
    curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
fi
if [ ! -f "train-labels-idx1-ubyte.gz" ]; then
    echo "Downloading MNIST training labels..."
    curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
fi

# Download test set
if [ ! -f "t10k-images-idx3-ubyte.gz" ]; then
    echo "Downloading MNIST test images..."
    curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
fi
if [ ! -f "t10k-labels-idx1-ubyte.gz" ]; then
    echo "Downloading MNIST test labels..."
    curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
fi

# Extract files
gunzip -f *.gz

cd ..

echo "Setup complete!"