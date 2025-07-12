#include "Neuron.h"

// c'tor
Neuron::Neuron(int numberInputs) {
  // Create a uniform distribution on range [-1, 1] to initialize the neuron
  // weights.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  // Initialize a random set of neuron weights
  mWeights.resize(numberInputs);
  for (size_t i = 0; i < mWeights.size(); ++i) {
    mWeights[i] = std::make_shared<Value>(Value(distribution(gen)));
  }

  // Initialize the bias to 0
  mBias = std::make_shared<Value>(0.0);
}

// Call function
std::shared_ptr<Value>
Neuron::call(const std::vector<std::shared_ptr<Value>> &x) const {
  // Verify the input vector is the correct length
  if (x.size() != mWeights.size()) {
    throw std::invalid_argument(
        "Vector of inputs must match number of weights in the neuron.");
  }

  // Compute dot product of inputs and weights/biases
  auto dot = mBias;
  for (size_t i = 0; i < x.size(); ++i) {
    dot = dot->operator+(mWeights[i]->operator*(x[i]));
  }

  // Apply hyperbolic tangent activation function
  return dot->tanh();
}

// Get Parameters function
std::vector<std::shared_ptr<Value>> Neuron::parameters() const {
  std::vector<std::shared_ptr<Value>> out = mWeights;
  out.push_back(mBias);
  return out;
}