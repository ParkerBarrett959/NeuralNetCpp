#include "Value.h"

// Addition Overload Operator
Value Value::operator+(const Value &other) const {
  // Create set of previous nodes
  std::set<std::shared_ptr<const Value>> prev = {
      std::make_shared<const Value>(other), shared_from_this()};

  // Create the new value
  Value out(this->data() + other.data(), prev);
  return out;
}

// Multiplication Overload Operator
Value Value::operator*(const Value &other) const {
  // Create set of previous nodes
  std::set<std::shared_ptr<const Value>> prev = {
      std::make_shared<const Value>(other), shared_from_this()};

  // Create the new value
  Value out(this->data() * other.data(), prev);
  return out;
}