# Genetic Algorithm AutoML Implementation

![Genetic Algorithm](https://img.shields.io/badge/License-MIT-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project implements a **Genetic Algorithm (GA)** to optimize the architecture and hyperparameters of a **Convolutional Neural Network (CNN)** for image classification tasks. By evolving a population of neural networks, the algorithm seeks the best combination of hyperparameters (such as the number of layers, filters, activation functions, optimizers, and loss functions) to achieve high accuracy on a given dataset.

## Features

- **Genetic Algorithm for Hyperparameter Optimization**: Automatically searches for the optimal CNN architecture and hyperparameters.
- **Dynamic Population Management**: Creates, evolves, and selects the best-performing networks over multiple generations.
- **Model Training and Evaluation**: Trains each network in the population and evaluates their performance using accuracy as the fitness metric.
- **Export Best Model**: Saves the best-performing model's architecture and weights for future use.
- **Visualization Tools**:
  - **Model Architecture Diagram**: Visual representation of the best model's architecture.
  - **Performance Metrics Plots**: Graphs showing the evolution of accuracy and loss over epochs.
- **Compatibility with Custom Datasets**: Easily switch between different datasets by adjusting input shapes and data preprocessing steps.

## Dataset

The project is configured to work with a custom dataset. Ensure your dataset matches the following structure:

- **Training Data**:
  - `x_train`: Shape `(5216, 256, 256, 3)` — 5216 RGB images of size 256x256.
  - `y_train`: Shape `(5216, 2)` — One-hot encoded labels for 2 classes.
- **Testing Data**:
  - `x_test`: Shape `(624, 256, 256, 3)` — 624 RGB images of size 256x256.
  - `y_test`: Shape `(624, 2)` — One-hot encoded labels for 2 classes.

> **Note**: Modify the `input_shape` and `num_classes` in the code if your dataset has different dimensions or number of classes.

## Installation

### Prerequisites

- Python 3.7 or higher
- [pip](https://pip.pypa.io/en/stable/) package manager

### Clone the Repository

```bash
git clone https://github.com/your-username/genetic-cnn-optimizer.git
cd genetic-cnn-optimizer
```

### Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install tensorflow keras numpy matplotlib
```

## Usage

### Configure the Dataset

Ensure your dataset is loaded correctly and assigned to the variables `x_train`, `y_train`, `x_test`, and `y_test`. The code assumes these variables are available for training and testing the models.

### Running the Genetic Algorithm
To execute the genetic algorithm and begin optimizing the CNN model, run:

```
# Example to start the GA process
from genetic_algorithm import GeneticAlgorithm

acc_goal = 0.99
GA = GeneticAlgorithm(population_size=5, mutation_rate=0.1, generations=10, epochs=5)
GA.evolve(acc_goal)
```

The above code will initialize the genetic algorithm with a population of CNNs and iteratively evolve them over 10 generations to reach the target accuracy.

## Contributing

Contributions to this project are welcome! If you have suggestions, bug fixes, or new features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

Special thanks to the open-source community and resources for Keras and TensorFlow, and to everyone contributing to the development of automated machine learning tools.