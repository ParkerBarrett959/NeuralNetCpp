#include "Neuron.h"

// c'tor
Neuron::Neuron(int numberInputs, const std::function<Value(double)> &activation)
    : mActivationFunction(activation) {
  // Create a uniform distribution on range [-1, 1] to initialize the neuron
  // weights.
  std::mt19937 generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  // Initialize a random set of neuron weights
  mWeights.resize(numberInputs);
  for (size_t i = 0; i < mWeights.size(); ++i) {
    mWeights[i] = Value(distribution(generator));
  }

  // Initialize the bias to 0
  mBias = Value(0.0);
}