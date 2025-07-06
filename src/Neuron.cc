#include "Neuron.h"

// c'tor
Neuron::Neuron(int numberInputs) {
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

// Call function
Value Neuron::call(const std::vector<Value> &x) const {
  // Verify the input vector is the correct length
  if (x.size() != mWeights.size()) {
    throw std::invalid_argument(
        "Vector of inputs must match number of weights in the neuron.");
  }

  // Compute dot product of inputs and weights/biases
  Value dot;
  for (size_t i = 0; i < x.size(); ++i) {
    Value xiwi = x[i] * mWeights[i];
    dot = dot + xiwi;
  }
  dot = dot + mBias;

  // Apply relu activation function
  // TODO: Add to Value
  return dot;
}