#pragma once

#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

#include "Value.h"

/**
 * Neuron
 *
 * @brief: A class which defines the neuron, or the core atomic unit of the
 * neural network framework. The neuron has a specific number of inputs, x1, x2,
 * ..., xn, weights w1, w2, ..., wn, and a bias term, b. The model of a neuron
 * is y = f(w1*x1 + w2*x2 + ... + wn*xn + b).
 */
class Neuron {
public:
  /**
   * c'tor
   *
   * @param numberInputs The integer number of inputs the neural net receives.
   */
  Neuron(int numberInputs);

  /**
   * Call Function
   *
   * @brief Run the neuron call function which computes the dot product of the
   * weights and baises with the input, and applies the activation function.
   *
   * @param x A vector of inputs, x1, x2, ..., xn
   *
   * @return A value result of the call operator on the neuron
   */
  Value call(const std::vector<Value> &x) const;

private:
  // Vector of neuron weights
  std::vector<Value> mWeights;

  // Neuron bias
  Value mBias;
};