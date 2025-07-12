#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include "../include/MultiLayerPerceptron.h"
#include "mnist_loader.h"

constexpr int inputSize = 28 * 28;
constexpr int hiddenSize = 64;
constexpr int outputSize = 10;
constexpr double learningRate = 0.01;
constexpr int epochs = 5;
constexpr int batchSize = 32;

std::pair<double, double> evaluate(
    const MultiLayerPerceptron& mlp,
    const std::vector<std::vector<double>>& images,
    const std::vector<int>& labels)
{
  int correct = 0;
  double totalLoss = 0.0;

  for (size_t i = 0; i < images.size(); ++i) {
    std::vector<std::shared_ptr<Value>> input;
    for (double pixel : images[i]) {
      input.push_back(std::make_shared<Value>(pixel));
    }

    auto output = mlp.call(input);

    // Compute softmax
    std::vector<double> expVals(outputSize);
    double sumExp = 0.0;
    for (int j = 0; j < outputSize; ++j) {
      expVals[j] = std::exp(output[j]->data());
      sumExp += expVals[j];
    }

    // Predicted class = argmax
    int prediction = std::distance(expVals.begin(),
      std::max_element(expVals.begin(), expVals.end()));
    if (prediction == labels[i]) {
      correct++;
    }

    double logProb = std::log(expVals[labels[i]] / sumExp);
    totalLoss += -logProb;
  }

  double avgLoss = totalLoss / images.size();
  double accuracy = static_cast<double>(correct) / images.size();
  return {avgLoss, accuracy};
}

int main() {
  std::vector<std::vector<double>> allImages;
  std::vector<int> allLabels;

  loadMNISTImages("../data/train-images-idx3-ubyte", allImages);
  loadMNISTLabels("../data/train-labels-idx1-ubyte", allLabels);

  std::cout << "Loaded " << allImages.size() << " images.\n";

  // Split into train/test (e.g., 80% train, 20% test)
  int totalSize = static_cast<int>(allImages.size());
  int trainSize = static_cast<int>(0.8 * totalSize);

  std::vector<int> indices(totalSize);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(std::random_device{}());
  std::shuffle(indices.begin(), indices.end(), rng);

  std::vector<std::vector<double>> trainImages, testImages;
  std::vector<int> trainLabels, testLabels;

  for (int i = 0; i < trainSize; ++i) {
    trainImages.push_back(allImages[indices[i]]);
    trainLabels.push_back(allLabels[indices[i]]);
  }

  for (int i = trainSize; i < totalSize; ++i) {
    testImages.push_back(allImages[indices[i]]);
    testLabels.push_back(allLabels[indices[i]]);
  }

  MultiLayerPerceptron mlp(inputSize, {hiddenSize, outputSize});
  auto parameters = mlp.parameters();

  // Evaluate before training
  auto [initialLoss, initialAcc] = evaluate(mlp, testImages, testLabels);
  std::cout << "Before training: Loss = " << initialLoss
            << ", Accuracy = " << initialAcc * 100.0 << "%\n";

  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::cout << "\nEpoch " << epoch + 1 << "/" << epochs << "\n";

    // Shuffle training data each epoch
    std::vector<int> trainIndices(trainImages.size());
    std::iota(trainIndices.begin(), trainIndices.end(), 0);
    std::shuffle(trainIndices.begin(), trainIndices.end(), rng);

    for (int i = 0; i < static_cast<int>(trainImages.size()); i += batchSize) {
      int end = std::min(i + batchSize, (int)trainImages.size());

      // Reset gradients
      for (auto &p : parameters) {
        p->setGradient(0.0);
      }

      double batchLoss = 0.0;

      for (int j = i; j < end; ++j) {
        int idx = trainIndices[j];

        std::vector<std::shared_ptr<Value>> input;
        for (double pixel : trainImages[idx]) {
          input.push_back(std::make_shared<Value>(pixel));
        }

        auto output = mlp.call(input);

        double sumExp = 0.0;
        for (const auto &o : output) {
          sumExp += std::exp(o->data());
        }

        auto label = trainLabels[idx];
        auto loss = std::make_shared<Value>(-std::log(std::exp(output[label]->data()) / sumExp));
        batchLoss += loss->data();
        loss->backward();
      }

      // Gradient descent
      for (auto &p : parameters) {
        double newVal = p->data() - learningRate * p->gradient();
        *p = Value(newVal);
      }

      std::cout << "Batch " << i / batchSize
                << ", Avg Loss = " << batchLoss / (end - i) << "\n";
    }

    // Evaluate after epoch
    auto [valLoss, valAcc] = evaluate(mlp, testImages, testLabels);
    std::cout << "After epoch " << epoch + 1 << ": "
              << "Loss = " << valLoss
              << ", Accuracy = " << valAcc * 100.0 << "%\n";
  }

  return 0;
}
