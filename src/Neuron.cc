#include "Neuron.h"

// c'tor
Neuron::Neuron(int numberInputs, const std::function<double()> &activation)
    : mActivationFunction(activation) {
  // Create a uniform distribution on range [-1, 1] to initialize the neuron
  // weights.
  std::mt19937 generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  // Initialize a random set of neuron weights
  mWeights.resize(numberInputs);
  for (auto &val : mWeights) {
    val = distribution(generator);
  }

  // Initialize the bias to 0
  mBias = 0.0;
}