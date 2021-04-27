#!/usr/bin/env bash
set -ex

ldconfig  # https://github.com/NVIDIA/nvidia-docker/issues/854

echo "Build Settings:"
echo "VERSION_SUFFIX:  $VERSION_SUFFIX"   # e.g. dev or ""
echo "PYTHON_VERSION:  $PYTHON_VERSION"   # e.g. 3.7

python3 -m pip install wheel twine 

# use separate directories to allow parallel build
BASE_BUILD_DIR=build/py$PYTHON_VERSION
python3 setup.py \
  build -b "$BASE_BUILD_DIR" \
  bdist_wheel -b "$BASE_BUILD_DIR/build_dist"
rm -rf "$BASE_BUILD_DIR" build torchshard.egg-info
