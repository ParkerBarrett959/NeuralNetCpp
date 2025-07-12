#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "../include/MultiLayerPerceptron.h"
#include "mNistLoader.h"

/**
 * Main MNIST Trainig Example
 */
int main() {
  std::vector<std::vector<double>> images;
  std::vector<int> labels;

  // Load training data
  loadMNISTImages("../data/train-images-idx3-ubyte", images);
  loadMNISTLabels("../data/train-labels-idx1-ubyte", labels);
  std::cout << "Loaded " << images.size() << " training images\n";

  // Define neural network with inputs equal to the size of the training images,
  // 2 hidden layers, and an output layer with 10 possible value [0 - 9]
  MultiLayerPerceptron mlp(28 * 28, {128, 64, 10});
  std::vector<std::shared_ptr<Value>> parameters = mlp.parameters();

  // Define training parameters
  const int epochs = 5;
  const double learningRate = 0.01;
  const int batchSize = 32;

  // Main training loop
  std::mt19937 rng(std::random_device{}());
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Print training epoch
    std::cout << "Epoch " << epoch + 1 << "\n";

    // Shuffle dataset
    std::vector<int> indices(images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Loop over images
    for (int i = 0; i < static_cast<int>(images.size()); i += batchSize) {
      int end = std::min(i + batchSize, (int)images.size());

      double loss = 0.0;

      // Reset gradients
      for (auto &p : parameters) {
        p->setGradient(0.0);
      }

      for (int j = i; j < end; ++j) {
        int idx = indices[j];

        // Create input Value vector
        std::vector<std::shared_ptr<Value>> input;
        for (double pixel : images[idx]) {
          input.push_back(std::make_shared<Value>(pixel));
        }

        auto output = mlp.call(input);

        // Softmax + NLL loss
        double sumExp = 0.0;
        for (const auto &o : output) {
          sumExp += std::exp(o->data());
        }

        auto label = labels[idx];
        auto correctLogProb = std::make_shared<Value>(
            -std::log(std::exp(output[label]->data()) / sumExp));
        loss += correctLogProb->data();

        correctLogProb->backward();
      }

      // SGD step
      for (auto &p : parameters) {
        double newVal = p->data() - learningRate * p->gradient();
        *p = Value(newVal);
      }
      std::cout << "Batch " << i / batchSize << " loss = " << loss / (end - i)
                << "\n";
    }
  }

  return 0;
}
