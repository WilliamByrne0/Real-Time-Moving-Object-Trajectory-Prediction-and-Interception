# Hexapod Robot Object Interception System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

A real-time object interception system using a hexapod robot (InsectBot Mini MKII) and a UR3e robotic arm. The system predicts moving object trajectories using Kalman filters, LSTMs, or Transformers and commands the robot to intercept.



## Features
- **Multiple Prediction Models**: Kalman Filter, LSTM, Transformer, Linear Motion Model.
- **Real-Time Vision**: HSV-based tracking with contour detection (customizable for different colors).
- **Robotic Control**: XML‑RPC communication with UR3e; custom inverse kinematics with collision avoidance.
- **Hexapod Integration**: Arduino firmware for InsectBot locomotion (forward, turn, reverse).
- **Visual Feedback**: OpenCV overlay showing predictions, intercept point, and FPS.
- **Configurable**: All parameters (camera, model, thresholds) are adjustable via a YAML config file.
- **Simulation Mode**: Test with recorded videos without hardware.

## System Overview
