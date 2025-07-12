# Script to Run Clang-Format Automatically

# Headers
clang-format -i example/*.h
clang-format -i include/*.h

# Source
clang-format -i example/*.cc
clang-format -i src/*.cc
