#pragma once

#include "Neuron.h"

/**
 * Layer
 *
 * @brief: A class which defines a a single layer of neurons in the neural
 * network.
 */
class Layer {
public:
  /**
   * c'tor
   *
   * @param numberInputs The integer number of inputs the layer receives.
   * @param numberOutputs The integer number of outputs the layer sends out.
   */
  Layer(int numberInputs, int numberOutputs);

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
  std::vector<Value> call(const std::vector<Value> &x) const;

private:
  // Vector of neurons in the layer
  std::vector<Neuron> mNeurons;
};