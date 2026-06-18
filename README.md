# Hexapod Robot Object Interception System

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

A real‑time interception system that tracks a moving hexapod robot (or any blue target) with a camera, predicts its future trajectory using Kalman filters, LSTMs, or Transformers, and commands a UR3e robotic arm to intercept it.

![Demo](docs/images/demo.gif) <!-- Replace with actual demo GIF -->

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Architecture](#software-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training Your Own Models](#training-your-own-models)
- [Robot Setup](#robot-setup)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The system processes live video from a webcam, extracts the blue object (the target), predicts its path using a chosen model, computes an intercept point that the UR3e arm can reach in time (using inverse kinematics and motion time estimation), and sends the arm to that point. The target hexapod is controlled separately via its Arduino firmware, allowing for dynamic testing.

## Features

- **Multiple Prediction Models**
  - Kalman Filter (constant velocity)
  - LSTM (sequence‑to‑sequence)
  - Transformer (self‑attention)
  - Linear (velocity smoothing)
- **Real‑Time Vision**
  - HSV colour filtering (blue)
  - Morphological cleaning and contour detection
  - Perspective warp to arena coordinates
- **Robotic Control**
  - XML‑RPC communication with UR3e
  - Custom inverse kinematics with collision avoidance
  - Latency measurement and compensation
  - Automatic selection of fastest joint solution
- **Visual Feedback**
  - Live video with overlay of predicted path, intercept point, and FPS
- **Simulation Mode**
  - Run on recorded videos without hardware

## Hardware Requirements

| Component | Details |
|-----------|---------|
| **Robotic Arm** | Universal Robots UR3e (or any with XML‑RPC and inverse kinematics) |
| **Target** | InsectBot Mini MKII hexapod (or any blue‑colored object) |
| **Camera** | 1080p webcam with manual exposure/white balance (recommended) |
| **Compute** | PC with Python 3.8+, OpenCV, PyTorch (CUDA optional) |
| **Network** | Ethernet or Wi‑Fi for UR3e communication |

## Software Architecture
Camera Feed
│
▼
┌─────────────────────────────────────┐
│ Tracking (HSV + Contours) │
│ Output: (x, y) in pixels │
└─────────────────────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ Prediction Engine │
│ - Kalman Filter │
│ - LSTM / Transformer (PyTorch) │
│ - Linear │
│ Output: List of predicted points │
└─────────────────────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ Interception Planner (Best_Search) │
│ - Iterate predicted points │
│ - Compute IK & travel time │
│ - Select feasible point │
│ Output: Intercept point (pixels) │
└─────────────────────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ Robot Interface (XML‑RPC) │
│ - Send pose to UR3e │
│ - Receive joint angles / status │
└─────────────────────────────────────┘
