#include "Value.h"

// Addition Overload Operator
std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value> &other) {
  // Create set of previous nodes
  std::vector<std::shared_ptr<const Value>> prev = {other, shared_from_this()};

  // Create the new value
  auto out = std::make_shared<Value>(mData + other->data(), prev);

  // Add backwards propagation function
  out->setBackward([out, selfPtr = shared_from_this(), other]() {
    selfPtr->setGradient(selfPtr->gradient() + out->gradient());
    other->setGradient(other->gradient() + out->gradient());
  });
  return out;
}

// Multiplication Overload Operator
std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value> &other) {
  // Create set of previous nodes
  std::vector<std::shared_ptr<const Value>> prev = {other, shared_from_this()};

  // Create the new value
  auto out = std::make_shared<Value>(mData * other->data(), prev);

  // Add backwards propagation function
  out->setBackward([out, selfPtr = shared_from_this(), other]() {
    selfPtr->setGradient(selfPtr->gradient() + other->data() * out->gradient());
    other->setGradient(other->gradient() + selfPtr->data() * out->gradient());
  });
  return out;
}

// Backward pass function
void Value::backward() {
  // Use const Value everywhere for consistency
  std::set<std::shared_ptr<const Value>> visited;
  std::vector<std::shared_ptr<const Value>> topoSort;

  // Lambda for topological sort
  std::function<void(std::shared_ptr<const Value>)> buildTopo =
      [&](std::shared_ptr<const Value> v) {
        if (!visited.count(v)) {
          visited.insert(v);
          for (const auto &child : v->previous()) {
            buildTopo(child);
          }
          topoSort.push_back(v);
        }
      };

  // Build topological sort
  buildTopo(shared_from_this());

  // Set gradient for the final node (self)
  mGrad = 1.0;

  // Run backprop in reverse topological order
  for (auto it = topoSort.rbegin(); it != topoSort.rend(); ++it) {
    // Cast away const to call the mutable backward function
    auto mutableVal = std::const_pointer_cast<Value>(*it);
    if (mutableVal->mBackward) {
      mutableVal->mBackward();
    }
  }
}