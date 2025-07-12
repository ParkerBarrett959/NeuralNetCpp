#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Read a big-endian int
inline int readInt(std::ifstream &ifs) {
  unsigned char bytes[4];
  ifs.read(reinterpret_cast<char *>(bytes), 4);
  return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

inline void loadMNISTImages(const std::string &path,
                            std::vector<std::vector<double>> &images) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open image file: " + path);
  }

  int magic = readInt(ifs);
  int numImages = readInt(ifs);
  int numRows = readInt(ifs);
  int numCols = readInt(ifs);

  images.resize(numImages, std::vector<double>(numRows * numCols));
  for (int i = 0; i < numImages; ++i) {
    for (int j = 0; j < numRows * numCols; ++j) {
      unsigned char pixel;
      ifs.read(reinterpret_cast<char *>(&pixel), 1);
      images[i][j] = pixel / 255.0; // Normalize
    }
  }
}

inline void loadMNISTLabels(const std::string &path, std::vector<int> &labels) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open label file: " + path);
  }

  int magic = readInt(ifs);
  int numLabels = readInt(ifs);
  labels.resize(numLabels);
  for (int i = 0; i < numLabels; ++i) {
    unsigned char label;
    ifs.read(reinterpret_cast<char *>(&label), 1);
    labels[i] = static_cast<int>(label);
  }
}
