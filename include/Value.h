#pragma once

#include <functional>
#include <memory>
#include <set>

/**
 * Value
 *
 * @brief: A class which defines a value type in the neural network for use in
 * reverse-mode automatic differentiation, aka backpropagation. The value type
 * holds scalar values and their gradients. Additionally, this class overloads
 * several common math operators for performing automatic differentiation.
 */
class Value : public std::enable_shared_from_this<Value> {
public:
  /**
   * c'tor
   *
   * @brief Set the scalar value and initializes the gradient to 0.
   *
   * @param val The scalar value (default = 0.0)
   * @param prev A vector of pointers to the previous (upstream) values (default
   * = empty set)
   */
  inline Value(double val = 0.0,
               const std::vector<std::shared_ptr<const Value>> &prev = {})
      : mData(val), mGrad(0.0), mPrevious(prev) {}

  /**
   * Data
   *
   * @brief Return the underlying data associated with the current value.
   *
   * @return The data stored in the value.
   */
  inline double data() const { return mData; }

  /**
   * Set Gradient
   *
   * @brief Set the underlying gradient value.
   *
   * @param grad The new value of the gradient
   */
  inline void setGradient(double grad) { mGrad = grad; }

  /**
   * Gradient
   *
   * @brief Return the underlying gradient associated with the current value.
   *
   * @return The gradient stored in the value.
   */
  inline double gradient() const { return mGrad; }

  /**
   * Set backward
   *
   * @brief Set the backward pass function
   *
   * @param func The backwards pass function
   */
  inline void setBackward(const std::function<void()> &func) {
    mBackward = func;
  }

  /**
   * Backward
   *
   * @brief Run the backward pass function
   */
  void backward();

  /**
   * Previous
   *
   * @brief Return the set of previous nodes
   */
  inline std::vector<std::shared_ptr<const Value>> previous() const {
    return mPrevious;
  }

  /**
   * Addition Overload Operator
   *
   * @brief Overloaded addition operator.
   *
   * @param other Another value to be added to the current one
   * @return Value with the data being the sum of the data from the two
   * contributing values.
   */
  std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &other);

  /**
   * Multiplication Overload Operator
   *
   * @brief Overloaded multiplication operator.
   *
   * @param other Another value to be multiplied by the current one
   * @return Value with the data being the product of the data from the two
   * contributing values.
   */
  std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &other);

private:
  // Current data
  double mData;

  // Current gradient
  double mGrad;

  // Backward pass function
  std::function<void()> mBackward;

  // The set of contributing/previous nodes in the net
  std::vector<std::shared_ptr<const Value>> mPrevious;
};