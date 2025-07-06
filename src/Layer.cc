#include "Layer.h"

// c'tor
Layer::Layer(int numberInputs, int numberOutputs) {
  // Loop over number of outputs and add neurons to layer
  for (int i = 0; i < numberOutputs; ++i) {
    mNeurons.push_back(Neuron(numberInputs));
  }
}

// Call operator overload
std::vector<Value> Layer::call(const std::vector<Value> &x) const {
  // Initialize output
  std::vector<Value> out;

  // Loop over neurons in layer and call operator
  for (const auto &neuron : mNeurons) {
    out.push_back(neuron.call(x));
  }
  return out;
}