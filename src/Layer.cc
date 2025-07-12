#include "Layer.h"

// c'tor
Layer::Layer(int numberInputs, int numberOutputs) {
  // Loop over number of outputs and add neurons to layer
  for (int i = 0; i < numberOutputs; ++i) {
    mNeurons.push_back(Neuron(numberInputs));
  }
}

// Call operator overload
std::vector<std::shared_ptr<Value>>
Layer::call(const std::vector<std::shared_ptr<Value>> &x) const {
  // Initialize output
  std::vector<std::shared_ptr<Value>> out;

  // Loop over neurons in layer and call operator
  for (const auto &neuron : mNeurons) {
    out.push_back(neuron.call(x));
  }
  return out;
}

// Get parameters of all neurons in layer
std::vector<std::shared_ptr<Value>> Layer::parameters() const {
  std::vector<std::shared_ptr<Value>> out;
  for (const auto &neuron : mNeurons) {
    auto p = neuron.parameters();
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}