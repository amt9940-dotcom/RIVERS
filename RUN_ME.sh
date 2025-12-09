#!/bin/bash

echo "================================================================================"
echo "EROSION SIMULATION - USING YOUR RIVERS NEW WEATHER"
echo "================================================================================"
echo ""
echo "This script runs the erosion simulation with YOUR existing weather system"
echo "from the 'Rivers new' code."
echo ""
echo "Files used:"
echo "  - Rivers new (your terrain, weather, layers)"
echo "  - erosion_simulation.py (erosion physics)"
echo "  - erosion_with_rivers_weather.py (integration)"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python3 erosion_with_rivers_weather.py

echo ""
echo "================================================================================"
echo "DONE!"
echo "================================================================================"
echo ""
echo "Output saved to: erosion_with_rivers_weather.png"
echo ""
