#include <iostream>

#include "Value.h"

// Main function
int main() {
  // Load in Training dataset
  // TODO...

  // Split dataset into training and evaluation portions

  // Initialize a neural network for training

  // Main Training Loop
  static constexpr int NUM_TRAINING_STEPS = 1000;
  for (int i = 0; i < NUM_TRAINING_STEPS; ++i) {
    // Forward prediction of loss
    double loss = 100.0;

    // Zero out gradients in network

    // Perform backpropagation

    // Apply stochastic gradient descent update

    // Print status
    std::cout << "Training Step: " << i << ", loss = " << loss << std::endl;
  }

  // Run evaluation dataset and display results
  // TODO...
  return 0;
}