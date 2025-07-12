#pragma once

#include "Layer.h"

/**
 * Multi-Layer Perceptron
 *
 * @brief: A class which defines a full multi-layer perceptron neural network.
 */
class MultiLayerPerceptron {
public:
  /**
   * c'tor
   *
   * @param numberInputs The integer number of inputs the layer receives.
   * @param numberOutputs A vector of the integer number of outputs in each
   * layer.
   */
  MultiLayerPerceptron(int numberInputs, const std::vector<int> &numberOutputs);

  /**
   * Call Function
   *
   * @brief Run the layer call function which runs the call function of each
   * neuron in the layer using the given inputs.
   *
   * @param x A vector of inputs, x1, x2, ..., xn
   *
   * @return A vector of value results of the call operator on each neuron
   */
  std::vector<std::shared_ptr<Value>>
  call(const std::vector<std::shared_ptr<Value>> &x) const;

  /**
   * Parameters
   *
   * @brief Get parameters of all neurons in layer
   */
  std::vector<std::shared_ptr<Value>> parameters() const;

private:
  // Vector of leyers in the neural network
  std::vector<Layer> mLayers;
};