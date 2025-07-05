#pragma once

/**
 * Value
 *
 * @brief: A class which defines a value type in the neural network for use in
 * reverse-mode automatic differentiation, aka backpropagation. The value type
 * holds scalar values and their gradients. Additionally, this class overloads
 * several common math operators for performing automatic differentiation.
 */
class Value {
public:
  /**
   * c'tor
   *
   * @brief Set the scalar value and initializes the gradient to 0.
   *
   * @param val The scalar value (default = 0.0)
   */
  inline Value(double val = 0.0) : mData(val), mGrad(0.0) {}

private:
  // Current data
  double mData;

  // Current gradient
  double mGrad;
};