#include "Value.h"

// Addition Overload Operator
std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value> &other) {
  // Create set of previous nodes
  std::set<std::shared_ptr<const Value>> prev = {other, shared_from_this()};

  // Create the new value
  auto out = std::make_shared<Value>(mData + other->data(), prev);
  return out;
}

// Multiplication Overload Operator
std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value> &other) {
  // Create set of previous nodes
  std::set<std::shared_ptr<const Value>> prev = {other, shared_from_this()};

  // Create the new value
  auto out = std::make_shared<Value>(mData * other->data(), prev);
  return out;
}