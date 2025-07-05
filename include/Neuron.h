#pragma once

#include <functional>
#include <random>
#include <vector>

/**
 * Neuron
 *
 * @brief: A class which defines the neuron, or the core atomic unit of the
 * neural network framework. The neuron has a specific number of inputs, x1, x2,
 * ..., xn, weights w1, w2, ..., wn, and a bias term, b. Additionally, the
 * neuron also holds an activation function, f. The model of a neuron is y =
 * f(w1*x1 + w2*x2 + ... + wn*xn + b).
 */
class Neuron {
public:
  /**
   * c'tor
   *
   * @param numberInputs The integer number of inputs the neural net receives.
   * @param activation The activation function
   */
  Neuron(int numberInputs, const std::function<double()> &activation);

private:
  // Vector of neuron weights
  std::vector<double> mWeights;

  // Neuron bias
  double mBias;

  // Neuron activation function
  std::function<double()> mActivationFunction;
};