```markdown
# Hexapod Robot Object Interception System

This repository contains the software stack for a real-time object interception system using a hexapod robot (InsectBot Mini MKII) and a UR3e robotic arm. The system uses computer vision, machine learning (LSTM and Transformer models), and classical control algorithms to predict and intercept moving objects.

## Hardware Requirements
- **Robotic Arm:** Universal Robots UR3e with XML-RPC interface
- **Hexapod Target:** InsectBot Mini MKII (or any blue object for tracking)
- **Camera:** 1080p webcam (manual exposure/white balance recommended)
- **Compute:** PC with Python 3.10+, optional NVIDIA GPU for AI models

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Weights:**
   Ensure you have the `models/` folder populated with the `.pth` and `.pt` weight files required by the prediction scripts.

## File Guide (`src/hexapod_intercept/`)

All source code is located in `src/hexapod_intercept/`. Here is what each file does:

### Main Execution & Control
- **`Obj_int_Real_06_05_26.py`**: The main script to run the interception system. It captures video, tracks the target, runs the chosen prediction model, calculates interception points via inverse kinematics, and sends commands to the UR3e robot.
- **`SendRobotXML.py`**: Sets up an XML-RPC server to communicate with the UR3e robot. It handles sending target poses, receiving joint angles, and measuring communication latency.
- **`IGM.py`**: Inverse Geometry Module. Contains the math for the UR3e forward and inverse kinematics. It calculates joint angles required to reach a specific 3D pose and filters out solutions that would collide with the table.
- **`urlib.py`**: Utility functions for UR robot communication (e.g., converting lists to robot pose formats).

### Computer Vision
- **`Tracking.py`**: Computer vision module. Uses HSV color filtering and contour detection to find the blue target object in the camera frame. Also contains the logic to automatically crop and warp the image to the arena boundaries using red tape detection.

### Prediction Models
- **`Prediction.py`**: Contains all prediction models:
  - `KalmanFilter`: A 2D constant-velocity Kalman filter for basic trajectory prediction.
  - `LSTM_Sim` / `Transformer_Sim`: Wrapper classes to load PyTorch models, normalize inputs, and output predicted future positions.
- **`Transformer_Train_03_05_2026.py`**: Script used to train the Transformer model using PyTorch. Includes dataset loading, training loops, and evaluation metrics (ADE/FDE).
- **`traindata_22_03_2026.py`**: Script to process raw video datasets. It extracts object centers, calculates velocities/accelerations, and saves the data in `.pt` format for training the AI models.

### Firmware
- **`insectbot_hexa.ino`**: Arduino code uploaded to the InsectBot Mini MKII to control hexapod locomotion.

## Usage

To run the live interception system:
1. Ensure the UR3e robot is powered on, connected to the network, and running the PolyScope XML-RPC program.
2. Position the camera so it can see the arena.
3. Run the main script:
   ```bash
   python src/hexapod_intercept/Obj_int_Real_06_05_26.py
   ```
4. A camera warm-up window will appear. Press **'W'** to compute the arena crop matrix, then press **'Q'** to start tracking and interception.

## Configuration
Parameters like camera resolution, FPS, model selection (Kalman/LSTM/Transformer), and timing thresholds are configured at the top of `Obj_int_Real_06_05_26.py`.
```
