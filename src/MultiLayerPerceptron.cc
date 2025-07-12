#include "MultiLayerPerceptron.h"

// c'tor
MultiLayerPerceptron::MultiLayerPerceptron(
    int numberInputs, const std::vector<int> &numberOutputs) {
  // Loop over number of outputs and add layers to network
  int inputSize = numberInputs;
  for (const auto &outputSize : numberOutputs) {
    mLayers.emplace_back(inputSize, outputSize);
    inputSize = outputSize;
  }
}

// Call operator overload
std::vector<std::shared_ptr<Value>>
MultiLayerPerceptron::call(const std::vector<std::shared_ptr<Value>> &x) const {
  // Initialize output
  std::vector<std::shared_ptr<Value>> out = x;

  // Loop over layer in network and call operator
  for (const auto &layer : mLayers) {
    out = layer.call(out);
  }
  return out;
}

// Get parameters of all neurons in network
std::vector<std::shared_ptr<Value>> MultiLayerPerceptron::parameters() const {
  std::vector<std::shared_ptr<Value>> out;
  for (const auto &layer : mLayers) {
    auto p = layer.parameters();
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}