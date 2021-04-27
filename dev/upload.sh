#!/usr/bin/env bash

twine check dist/*
twine upload --verbose --repository testpypi dist/*
