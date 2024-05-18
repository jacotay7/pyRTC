#!/bin/bash

# This script builds the Sphinx documentation and then runs a live server for development.

echo "Building the HTML documentation with Sphinx..."
make html

echo "Starting the live-reloading Sphinx server..."
sphinx-autobuild . _build/html
