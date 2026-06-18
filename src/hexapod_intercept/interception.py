import sys
import cv2 as cv
import numpy as np
import time

import Prediction
import Tracking as T
import SendRobotXML as send
from IGM import IGM_Fastest_sol, vec2rot

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1280,720  # Arena width = 0.47 m, height = 0.12 m, aspect ratio = 3.917  |   1280 / 3.917 = 327 pixels
BALL_RADIUS = 15
TARGET_COLOR = (0, 255, 0)
FOLLOWER_COLOR = (255, 0, 0)
grad_list = []

SIMULATION = 0           # If False, uses webcam input else use pre-recorded video
SEND_XMLRPC = 1          # Send interception positions to the robot
FILTER = "blue"             # "blue" for blue, anything else based on light

LSTM = 0
TRANSFORMER = 0
KALMANFILTER = 1
LINEARMODEL = 0

OCCLUSIONS = 0 # Works poorly with AI models
KALMANFILTER_CENTRE = 0 # Causes increased error with AI models, not trained
LINEAR_SMOOTH = 60       # number of frames to average velocity over
MAX_UNCERTAINTY = 9999999990       # Uncertainty threshold for Kalman Filter, it means the object could be anywhere within a circle of radius 50 pixels, adjust based on the scale of your environment and acceptable error
FPS = 25.223        #25.223 #30
dt = 1/25.223       # second per frame
VideoSpeed = 1     #int(round((dt/FPS )* 1000)) # Convert to ms for waitKey
REAL_SPEED = 1
MIN_TIME =  2 # seconds
MAX_TIME = 99 # seconds
ERROR_THRESH = 1 # Error threshold for auto intercept      85 good for transformer
SHORT_ERROR = 100 # only check N frames for error, 150 takes alot of time
GRAD_THRESH = 0.50#0.5#0.3#0.01#0.1#0.05   # 0.04 when testing transformer and LSTM,     0.1 for KF and    0.2/0.1 linear
DEBUG = 0
PREDICTION_LENGTH = 150
VIDEO_PATH = r"Datasets\Bug_walk_8_03_2026\Cropped\bug (55)_cropped.mp4" # 18 vertical, 37 horizontal, 55 both, 7 
OUTPUT_PATH = r"Datasets\Bug_walk_8_03_2026\Output\bug (60)_output_v1.mp4"
SAVE_VIDEO = 0
# ----------------------------------


# if LINEARMODEL:
#     GRAD_THRESH = 0.1  #0.5
    
def PixelToPoseX(pixel):
    x_start = -0.010812152843194799
    x_end   = 0.46
    return x_start + (pixel / (WIDTH - 1)) * (x_end - x_start)

def PixelToPoseY(pixel):
    y_start = -0.19926
    y_end   = -0.32166
    return y_start + (pixel / (HEIGHT - 1)) * (y_end - y_start)

def Point_To_Angles(x,y):
    x = PixelToPoseX(x)
    y = PixelToPoseY(y)
    # x  y     z      rx      ry     rz u = [ x, y, 0.0, -0.125, 0.005, -1.513]
    #u = [ x, y, 0.005, -0.125, 0.005, np.pi/2] # z = -0.005 for almost touching table
    u = [x, y, 0.100, 2.28, -2.161, 0] 
    sol = IGM_Fastest_sol(u,REST_ANGLE)
    #print("Target q = ",sol)
    return sol

def Waypoint_Time(rest_angle, target_angles):
    rest_array   = np.array(rest_angle)
    target_array = np.array(target_angles)
    
    qmax         = (32/180) * np.pi   # rad/s
    acceleration = (100/180) * np.pi  # rad/s²

    distance_q = rest_array - target_array
    distance_q_shortest = (distance_q + np.pi) % (2 * np.pi) - np.pi
    distance_q_shortest = np.abs(distance_q_shortest)
 
    # Time and distance to ramp up to qmax (and back down)
    time_acc     = qmax / acceleration          # time for one accel/decell
    distance_acc = 0.5 * acceleration * time_acc**2  # distance for one accel/decell

    times = []
    for dist in distance_q_shortest:

        # long move, reaches qmax, trapezoid
        if dist >= 2 * distance_acc:
            distance_cruise = dist - 2 * distance_acc
            time_cruise     = distance_cruise / qmax
            total_time      = 2 * time_acc + time_cruise

        # short move, never reaches qmax, triangle
        else:
            v_peak = np.sqrt(acceleration * dist)
            t_peak = v_peak / acceleration
            total_time = 2 * t_peak         # ramp up + ramp down

        times.append(total_time)

    return max(times) # return the largest time to point for all joints


def Best_Search(Predicted_Path,x,y):
    # TIME IN FRAMES NOT SECONDS
    Time = []
    t_start = time.perf_counter()
    Time_diff = 0
    step = 0
    denom = 2

    # if SEND_XMLRPC:
    #     measured_latency = send.get_latency() + 0.06 # calculating inverse kinematics
    # else:
    #     measured_latency = 0.0

    measured_latency = 0.064

    print("measured_latency CALCULAATION: ",measured_latency)

    while step < len(Predicted_Path):

        Px,Py = Predicted_Path[step] # Predicted point of the target at this step

        #Filter out of bounds
        if Px < 0 or Px > WIDTH:
            step += 1
            continue
        if Py < 0  or Py > HEIGHT:
            step += 1
            continue

        # calculate IGM joint positions
        current_angles = Point_To_Angles(Px,Py)

        if current_angles is None:
            step += 1
            continue

        # Time to Predicted point
        # add an extra +1 takes 30-38ms to calculate, and step starts from 0
        #TimeToPoint_T = step + 1  #Time to each point in pixel/frame
        
        time_to_point_sec = Waypoint_Time(REST_ANGLE, current_angles)
        

        t_target = (step + 1) * dt  # seconds until target reaches this prediction step
        t_robot = time_to_point_sec + measured_latency # total time robot needs (kinematics + communication)
        Time_diff = t_target - t_robot # Error between when the target will be at the point and when the follower can reach that point
        #print("Time_diff",Time_diff)

        if Time_diff < MIN_TIME: # if the robot wont make it in time then skip ahead frames
            skip = int((MIN_TIME - Time_diff) / (dt * denom)) + 1
            step += skip 
            denom *= 1.4 # Decrease search exponentially
            print(f"Skipped {skip} frames")
            continue
            
        if MIN_TIME <= Time_diff <= MAX_TIME:
            point = np.array([Px, Py])
            elapsed = (time.perf_counter() - t_start) * 1000
            print(f"Best_Search found match in {elapsed:.2f} ms (step {step})")
            return [t_target, t_robot, Time_diff], point
        
        step += 1
        
                
        #     if step > 0:
        #         prev = Predicted_Path[step - 1]
        #         grad_x = Px - prev[0]
        #         grad_y = Py - prev[1]
        #     else:
        #         grad_x = Px - x
        #         grad_y = Py - y

        #     dist = math.sqrt(grad_x**2 + grad_y**2)

        #     if dist > 0:
        #         normx = grad_x / dist
        #         normy = grad_y / dist

        #         Pix_map_x = (WIDTH - 1) / ((0.46 - (-0.010812152843194799)) * 1000)
        #         Pix_map_y = (HEIGHT - 1) / ((0.32166 - 0.19926) * 1000)
        #         offset_mm = 100 # 70 mm offset in direction of hexa bug
        #         offset_pix_x = Px + normx * offset_mm * Pix_map_x
        #         offset_pix_y = Py + normy * offset_mm * Pix_map_x

        #     else:
        #         offset_pix_x, offset_pix_y = Px, Py

            # print(f"--- Intercept Debug ---")
            # print(f"Bug current pos:      x={x:.1f}, y={y:.1f}")
            # print(f"Intercept point:      Px={Px:.1f}, Py={Py:.1f}")
            # print(f"Direction vector:     grad_x={grad_x:.2f}, grad_y={grad_y:.2f}")
            # print(f"Unit normal:          normx={normx:.3f}, normy={normy:.3f}")
            # print(f"Pixels per mm (X/Y):  {Pix_map_x:.3f}, {Pix_map_y:.3f}")
            # print(f"Offset mm:            {offset_mm} mm")
            # print(f"Offset in pixels:     dx={normx * offset_mm * Pix_map_x:.1f}, dy={normy * offset_mm * Pix_map_y:.1f}")
            # print(f"Offset point:         x={offset_pix_x:.1f}, y={offset_pix_y:.1f}")
            # print(f"----------------------")
            
            #offset_point = np.array([offset_pix_x, offset_pix_y])
            
    if Time_diff:
        print("Last predicted point time diff",Time_diff)
    print("No Time Found")
    return None


if __name__ == "__main__":
    # variables
    x2, y2 = WIDTH // 2, 0
    vx2 = 0
    vy2 = 0
    print_atpoint = False
    tti_start = 0
    Tti = 0
    is_intercepting = False
    centers = []
    Save_Path = []
    Predicted_Path = []
    PredictedPoints = []
    PrevCenters = None
    Currentcenters = False
    timing_robot = False
    VelocityPrev = np.array([0,0])
    Errorlist = []
    mean_mm_list = []
    frame_count = 0
    fps = 0
    full_frame_time = 0

    robot_was_moving = False     
    robot_start_time = 0.0            
    actual_move_time_s = 0.0
    predicted_move_time_s = 0.0

    #--Tracking and XML-RPC--
    if SEND_XMLRPC:
        #Server send init
        send.XMLServer()
        REST_ANGLE = send.get_angle()
        print("Rest Angle Recieved: ", REST_ANGLE)
        # [-3.141967121753486, -1.521974054431098, 1.855550591143842, -1.904451509513785, -1.57098704973332, 0.001118845075076917]

    else:
        # Test rest position
        test = [213.4,270,123.8,-121.52,-92.2,58.55]
        test = np.deg2rad(test)
        
        # u = [PixelToPoseX(x2),PixelToPoseY(y2),0.5,   np.pi/2,0,0]
        u = [PixelToPoseX(x2), PixelToPoseY(y2), 0.200, 2.28, -2.161, 0]
        # Rest_q = np.zeros(6)

        # test = np.zeros(6)
        REST_ANGLE = IGM_Fastest_sol(u,test)
        print(REST_ANGLE)
        

    tracker = T.Tracking()
    matrix = None

    #--Kalman Filter--
    if KALMANFILTER or KALMANFILTER_CENTRE:
        kf_target = Prediction.KalmanFilter()
        kf_target.x = np.array([[0],
                         [0],
                         [0],
                         [0]])
    
    if LINEARMODEL:
        velocity_history = []

    #--AI Prediction--
    if LSTM:
        import torch
        model = Prediction.LSTM_Sim()
        train_data        = torch.load("models/metadata.pt", weights_only=False)
        HISTORY_LENGTH    = train_data['History_length']
        PREDICTION_LENGTH = train_data['Prediction_length'] #* AUTO_REGRESSION
        input_features    = train_data['Features']
        Output_dim        = train_data['Output']
        vel_history = [[0.0, 0.0]] * HISTORY_LENGTH
        acc_history = [[0.0, 0.0]] * HISTORY_LENGTH
        feature_history = [[0.0,0.0,0.0,0.0,0.0,0.0]] * HISTORY_LENGTH

    if TRANSFORMER:
        import torch
        model = Prediction.Transformer_Sim()
        train_data        = torch.load("models/metadata.pt", weights_only=False)
        HISTORY_LENGTH    = train_data['History_length']
        PREDICTION_LENGTH = train_data['Prediction_length'] #* AUTO_REGRESSION
        input_features    = train_data['Features']
        Output_dim        = train_data['Output']
        vel_history = [[0.0, 0.0]] * HISTORY_LENGTH
        acc_history = [[0.0, 0.0]] * HISTORY_LENGTH
        feature_history = [[0.0,0.0,0.0,0.0,0.0,0.0]] * HISTORY_LENGTH

    #--------

    if SIMULATION:
        Vid = cv.VideoCapture(VIDEO_PATH)
        fps_video = Vid.get(cv.CAP_PROP_FPS)
        print(f"Video FPS: {fps_video}")
        
    else:
        Vid = cv.VideoCapture(1)
        Vid.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)      # 1 = manual, 3 = auto (varies by driver)
        Vid.set(cv.CAP_PROP_EXPOSURE, -6)          # Manual exposure value (negative = shorter)
        Vid.set(cv.CAP_PROP_AUTO_WB, 0)            # Disable auto white balance
        Vid.set(cv.CAP_PROP_WB_TEMPERATURE, 4600) # Set manual white balance
        Vid.set(cv.CAP_PROP_AUTOFOCUS, 0)          # Disable autofocus
        Vid.set(cv.CAP_PROP_FOCUS, 0)             # Set manual focus value
        Vid.set(cv.CAP_PROP_GAIN, 0)              # Manual gain
        Vid.set(cv.CAP_PROP_BRIGHTNESS, 128)      # Manual brightness
        Vid.set(cv.CAP_PROP_CONTRAST, 128)        # Manual contrast
        
        while True: #Camera warmup
            ret, frame = Vid.read()

            if ret is None:
                print("Failed to read frame from video.")
                continue
            
            
            frame = cv.resize(frame,(WIDTH,HEIGHT))
            ori = frame.copy()

            cv.putText(ori, "Controls:", (10, 25),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(ori, "W - Compute matrix", (10, 50),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.putText(ori, "Q - Start tracking", (10, 75),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.imshow("original", ori)

            key = cv.waitKey(1) & 0xFF
            if key == ord('w'):
                
                # Test IMG of areana
                # frame = cv.imread("Object Prediction/Cropping Arena/arena (6).jpg")
                
                # Return matrix to crop arena from camera feed
                matrix = tracker.Crop_Arena(frame,WIDTH,HEIGHT)

                if matrix is not None:
                    frame = cv.warpPerspective(frame, matrix, (WIDTH, HEIGHT), flags=cv.INTER_CUBIC)
                else:
                    print("No Areana Box Found")

                cv.putText(frame, f"Press Q to start", (0, 20),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv.imshow("frame", frame)
            
            if key == ord('q'):
                cv.destroyAllWindows()
                break

    # Error calulation
    Error_centres = []
    Error_prediction = []
    mean_error = np.inf
    prev_avg = 0
    grad = np.inf
    predicted_frame = False

            
    # Timer for FPS Calculation
    fps_list = []
    fps_time = 0

    # Video output
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    if SAVE_VIDEO:
        out = cv.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))
    Object_Status = False
    while True:
        fps_time = time.perf_counter()
        ret, frame = Vid.read()
        fps_python = time.perf_counter()
        
        
        if not ret: # Video end
            print("ret error")
            #Vid.set(cv.CAP_PROP_POS_FRAMES, 0)
            #continue
            break

        frame_count += 1
        frame = cv.resize(frame,(WIDTH,HEIGHT))

        # Warp to areana only when not simulating
        if not SIMULATION:
            if matrix is not None:
                frame = cv.warpPerspective(frame, matrix, (WIDTH, HEIGHT), flags=cv.INTER_CUBIC)

            else:
                print("No Areana Box Found")
        else:
            frame = cv.resize(frame, (WIDTH, HEIGHT), interpolation=cv.INTER_AREA)

        if SEND_XMLRPC:
            robot_stop = send.is_robot_stopped()
            if not robot_stop: #check if robot is moving

                if timing_robot:
                    time_send_move_ms = (time.perf_counter() - send_timer ) * 1000.0
                    print("Robot move command ms: ",time_send_move_ms)
                    timing_robot = False
                    
                    robot_start_time = time.perf_counter() # Actual robot time move

                robot_was_moving = True
                print("\rRobot is moving", end="", flush=True)
                cv.imshow("frame", frame)
                if cv.waitKey(VideoSpeed) & 0xFF in (ord('q'), 27):
                    break
                continue

        if robot_was_moving:
            # first frame after motion ended
            arrival_time = time.perf_counter()
            actual_move_time_s = arrival_time - robot_start_time
            print(f"Robot stopped. Actual move time: {actual_move_time_s*1000:.1f} ms")
            robot_was_moving = False   # reset the flag

            if SEND_XMLRPC:
                predicted_ms = t_robot * 1000.0                  # from Best_Search
                actual_ms = actual_move_time_s * 1000.0
                error_ms = actual_ms - predicted_ms
                print(f"Predicted robot travel (kin + latency): {predicted_ms:.1f} ms")
                print(f"Actual robot travel:                    {actual_ms:.1f} ms")
                print(f"Prediction error:                       {error_ms:+.1f} ms")
                print(f"Early‑arrival (target ‑ robot):         {t_diff*1000:.1f} ms")

        #Make sure there are point in Currentcenters
        if Currentcenters: 
            PrevCenters = Currentcenters

        timer_contour = time.perf_counter()
        Currentcenters, mask = tracker.Contours(frame,FILTER,DEBUG)
        elapsed_cont = time.perf_counter() - timer_contour

        # Handle object tracking failing
        if len(Currentcenters) == 0:
            if PrevCenters == None:
                print("\rNo Initial Object Found", end= "")
                Object_Status = False

                cv.imshow("frame",frame)
                if cv.waitKey(VideoSpeed) & 0xFF in (ord('q'), 27):
                    break

                if DEBUG:
                    cv.imshow("No Points", mask)
                    cv.imshow("frame",frame)
                    if cv.waitKey(VideoSpeed) & 0xFF in (ord('q'), 27):
                        break
                continue

            
            # Use Predicted_Path positions for occlusions
            elif len(Predicted_Path) > 0 and OCCLUSIONS:
                #Currentcenters = PrevCenters
                pred_x, pred_y = Predicted_Path[0]
                Currentcenters = [[float(pred_x), float(pred_y)]] # need to convert numpy float to python float
                predicted_frame = True # Used for error calulations, ignore when using predicted frame as centre

                print(f"No Points Found — using predicted position ({pred_x:.1f}, {pred_y:.1f})")
                
                if DEBUG:
                    cv.circle(frame, (int(pred_x), int(pred_y)), BALL_RADIUS, (0, 165, 255), 2)
                    cv.imshow("No Points", mask)
                    cv.imshow("frame",frame)
                    if cv.waitKey(VideoSpeed) & 0xFF in (ord('q'), 27):
                        break

            else:
                # No detection and no prediction available skip frame
                print("No Points Found and no prediction available, skipping frame")
                cv.imshow("frame", frame)
                if cv.waitKey(VideoSpeed) & 0xFF in (ord('q'), 27):
                    break
                continue
            
        else:
            predicted_frame = False
        

            if Object_Status == False: # Only print once
                Object_Status = True
                print("\nObject Found!")
        
        centers.append(Currentcenters[0])
        x,y = Currentcenters[-1]

        #Debug Colour tracking
        if DEBUG:
            cv.imshow("Mask", mask)
            #cv.imshow("Frame",frame)
            if cv.waitKey(VideoSpeed) & 0xFF in (ord('q'), 27):
                break

        #Draw the target and follower
        cv.circle(frame, (int(x), int(y)), BALL_RADIUS, TARGET_COLOR, -1)
        cv.circle(frame, (int(x2), int(y2)), BALL_RADIUS, FOLLOWER_COLOR, -1)

        if KALMANFILTER_CENTRE:
            # Kalman Filter predict and update loop
            kf_target.time = dt
            kf_target.Predict()
            kf_target.Update((x,y))
            
            Predicted_Path = []
            future = kf_target.x.copy()
            P_future = kf_target.P.copy()

            x,y = future[:2].flatten()
            

        if KALMANFILTER:
            # Kalman Filter predict and update loop
            kf_target.time = dt
            kf_target.Predict()
            kf_target.Update((x,y))
            
            Predicted_Path = []
            future = kf_target.x.copy()
            P_future = kf_target.P.copy()
            
            for _ in range(PREDICTION_LENGTH):
                future = kf_target.TransitionModel @ future
                P_future = kf_target.TransitionModel @ P_future @ kf_target.TransitionModel.T + kf_target.Q
                
                uncertainty = np.sqrt(P_future[0,0] + P_future[1,1])
                
                if uncertainty > MAX_UNCERTAINTY:
                    break  # stop predicting when too uncertain
                
                Predicted_Path.append(future[:2].flatten())  # Store predicted position (x,y)


        timer_ai = time.perf_counter()
        #AI and ML
        if LSTM or TRANSFORMER:
            #Points to Velocity
            cur = np.array(Currentcenters) # Remove []
        
            if PrevCenters != None:
                Velocity = cur[-1] - PrevCenters[-1]
                Acceleration = Velocity - VelocityPrev
                VelocityPrev = Velocity.copy()


                features = np.array([   x,                  y,
                                        Velocity[0],        Velocity[1],
                                        Acceleration[0],    Acceleration[1]])
                
                
                
                feature_history.append(features.tolist())

                if len(feature_history) > HISTORY_LENGTH:
                    feature_history.pop(0)

                features = np.array(feature_history)  # (HISTORY_LENGTH, 7)
                features = features.reshape(1, HISTORY_LENGTH, -1) # (1, HISTORY_LENGTH, 7)
                
                Predicted_Velocites = model.predict_point(features)
                
                # Get predicted x,y Positions
                # Get the cumulative sum and add this to x,y to get predicted positions of x,y at each time step
                Predicted_x = Predicted_Velocites[0,:,0].cumsum() + x
                Predicted_y = Predicted_Velocites[0,:,1].cumsum() + y
                
                Predicted_Path = np.column_stack((Predicted_x, Predicted_y))  
                Predicted_Path = Predicted_Path.reshape(PREDICTION_LENGTH, 2)
                
                # for i in range(AUTO_REGRESSION):
                #     # Last predicted position
                #     x_next = Predicted_Path[-1, 0]
                #     y_next = Predicted_Path[-1, 1]

                #     pred_vels = Predicted_Velocites[0]   # (PRED_LEN,2)

                #     # --- compute acceleration ---
                #     concat_vel = np.vstack([vel_prev, pred_vels])
                #     pred_acc = np.diff(concat_vel, axis=0)

                #     # --- build feature matrix ---
                #     features = np.hstack([
                #         pred_vels,
                #         pred_acc
                #     ])   # (PRED_LEN,2)

                #     # reshape for model
                #     features = features.reshape(1, PREDICTION_LENGTH, -1)

                #     # predict next velocities
                #     Predicted_Velocites = model.predict_point(features)

                #     # cumulative sum to get each position
                #     Predicted_x = Predicted_Velocites[0,:,0].cumsum() + x_next
                #     Predicted_y = Predicted_Velocites[0,:,1].cumsum() + y_next

                #     Path = np.column_stack((Predicted_x, Predicted_y)) # Current path
                #     Predicted_Path = np.vstack((Predicted_Path, Path)) # Total path

                #     # update for next loop
                #     vel_prev = pred_vels[-1]
    
        elapsed_ai = time.perf_counter() - timer_ai


        if LINEARMODEL:
            if PrevCenters is not None:
                vx = x - PrevCenters[-1][0]
                vy = y - PrevCenters[-1][1]
                
                velocity_history.append((vx, vy))
                if len(velocity_history) > LINEAR_SMOOTH:
                    velocity_history.pop(0)

                # Average velocity over last N frames
                avg_vx = sum(v[0] for v in velocity_history) / len(velocity_history)
                avg_vy = sum(v[1] for v in velocity_history) / len(velocity_history)

                Predicted_Path = []
                for step in range(1, 1 + PREDICTION_LENGTH):
                    px = x + avg_vx * step
                    py = y + avg_vy * step
                    Predicted_Path.append(np.array([px, py]))


        # Make sure it is a real position for error calculations
        if not predicted_frame and PrevCenters != None:
            PredictedPoints.append(Predicted_Path)  # -1, or -2 play around with it -2 seems lower error???
            Error_centres.append((x,y))

        # Error Calculation
        if len(Error_centres) > SHORT_ERROR:#       PREDICTION_LENGTH + PREDICTION_LENGTH*AUTO_REGRESSION:
            # centers_np = np.array(Error_centres[1:]) # Discard first position, this is the base position
            # Predict_np = np.array(PredictedPoints[0]) # First 30 predictions from the base position
            # Error = np.linalg.norm(centers_np - Predict_np, axis=1) # Distance every point is away from the actual path
            # mean_error = np.mean(Error) # Mean of all errors

            centers_np = np.array(Error_centres[1:SHORT_ERROR + 1])
            Predict_np = np.array(PredictedPoints[0])[:SHORT_ERROR]
            Error = np.linalg.norm(centers_np - Predict_np, axis=1)
            mean_error = np.mean(Error)

            Errorlist.append(mean_error) 
            Error_centres.pop(0)
            PredictedPoints.pop(0)

            act_m  = np.array([[PixelToPoseX(p[0]), PixelToPoseY(p[1])] for p in centers_np])
            pred_m = np.array([[PixelToPoseX(p[0]), PixelToPoseY(p[1])] for p in Predict_np])

            act_mm  = act_m  * 1000.0
            pred_mm = pred_m * 1000.0

            Error_mm = np.linalg.norm(act_mm - pred_mm, axis=1)
            mean_error_mm = np.mean(Error_mm)
            mean_mm_list.append(mean_error_mm)

        
        # Display the total path
        if len(Predicted_Path) > 0:
            pts = np.array(Predicted_Path, dtype=np.int32).reshape((-1, 1, 2))
            cv.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=4)

        # Display the predicted path when intercept started
        if len(Save_Path) > 0:
            for  px, py in Save_Path:
                cv.circle(frame, (int(px), int(py)), 2, (255/2, 255/2, 0), -1)

            #Draw The Point Chosen
            Ix = intercept[0].item()
            Iy = intercept[1].item()
                
            cv.line(frame, (int(x2), int(y2)), (int(Ix), int(Iy)), (255, 255, 0), 1)
            cv.circle(frame, (int(Ix), int(Iy)), 3,  (255, 0, 0), -1)    

        if predicted_frame:
            grad = np.inf # grad inf becasue no real measurement
        
        else:
            if Errorlist:
                avgError = np.mean(Errorlist[-10:])
                grad = avgError - prev_avg
                prev_avg = avgError
                #print("mean error = ",avgError)
                #print("grad = ",grad)
                #grad_list.append(abs(grad))
                #print(f"min grad = {min(grad_list)}")
            else:
                avgError = np.inf
                grad = np.inf
        
        #if frame_count % 5:
            #print (f"Gradient of error: {grad} | Mean of error: {avgError}")
                # 0.07 for linear
                #0.04 for LSTM transformer

        if (abs(grad) < GRAD_THRESH) and not is_intercepting: #Start Intercept Prediction with Space bar
            
            result = Best_Search(Predicted_Path,x,y) 
            
            if result == None:
                print("\nNo Time Solution Found!")

            else:
                Save_Path = Predicted_Path
                ((t_target, t_robot, t_diff), intercept) = result # time in seconds
                print(f"Target arrival in {t_target:.3f}s | Robot needs {t_robot:.3f}s | Difference {t_diff:.3f}s")
                intercept_start_time = time.perf_counter()

                print_atpoint = True # print once

                is_intercepting = True
                
                # Send Point To UR Robot
                if SEND_XMLRPC:
                    send_timer = time.perf_counter()
                    timing_robot = True
                    send.UpdatePoints(intercept[0],intercept[1])
                    print(f"Sent {intercept[0],intercept[1]} Predicted points ") 
        
            

        if is_intercepting:
            
            elapsed = time.perf_counter() - intercept_start_time

            if elapsed >= t_robot and not print_atpoint:
                print("Robot should be at intercept point now")
                print_atpoint = True


            if elapsed >= t_target:
                is_intercepting = False
                Intercepted = True # Turn off auto intercept once intercepted
                
                overlay = frame.copy()
                
                cv.putText(frame, "INTERCEPT FINISHED, Press any key to continue", 
                           (WIDTH//4 + 20, HEIGHT//2 + 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                cv.imshow("frame", frame)
                
                # 3. PAUSE until any key is pressed
                print("Simulation Paused. Press any key in the OpenCV window to continue.")
                cv.waitKey(0)


                print(f"Distance from centre and predicted: x = {x-intercept[0]} y = {y-intercept[1]}")

                # Reset all variables
                is_intercepting = False
                PrevCenters = None
                Save_Path = []
                Predicted_Path = []
                PredictedPoints = []
                centers = []
                mean_mm_list = []
                Error_centres = []
                Error_prediction = []
                Errorlist = []
                mean_error = np.inf
                prev_avg = 0
                grad = np.inf
                VelocityPrev = np.array([0, 0])
                predicted_frame = False
                Object_Status = False
                Currentcenters = False
                
                if LSTM or TRANSFORMER:
                    vel_history  = [[0.0, 0.0]] * HISTORY_LENGTH
                    acc_history  = [[0.0, 0.0]] * HISTORY_LENGTH
                    feature_history = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * HISTORY_LENGTH

                if KALMANFILTER:
                    kf_target.x = np.array([[0], [0], [0], [0]])
                    kf_target.P = np.eye(4) * 500

                if LINEARMODEL:
                    velocity_history = []

                print("Reset complete. Ready for next intercept.")
                continue  


            # tti_frame_counter += 1
            # if tti_frame_counter > Tti_frames_F - LATENCY_FRAMES: # If reached the time to intercept stop moving
            #     cv.putText(frame, "ROBOT AT POSITION", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            #     print("Follower Reached Point")
            

            # if tti_frame_counter > Tti_frames_T: # If reached the time to intercept stop
            #     is_intercepting = False
            #     Intercepted = True # Turn off auto intercept once intercepted
                
            #     overlay = frame.copy()
            #     cv.rectangle(overlay, (WIDTH//4, HEIGHT//2 - 50), (3*WIDTH//4, HEIGHT//2 + 50), (0,0,0), -1)
            #     cv.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
            #     cv.putText(frame, "INTERCEPT FINISHED. Press any key to continue...", 
            #                (WIDTH//4 + 20, HEIGHT//2 + 10), 
            #                cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            #     # 2. Force an update to the window so you can see the message
            #     cv.imshow("frame", frame)
                
            #     # 3. PAUSE until any key is pressed
            #     print("Simulation Paused. Press any key in the OpenCV window to continue.")
            #     cv.waitKey(0)

            #     # Reset all variables
            #     is_intercepting = False
            #     Save_Path = []
            #     Predicted_Path = []
            #     PredictedPoints = []
            #     centers = []
            #     Error_centres = []
            #     Error_prediction = []
            #     Errorlist = []
            #     mean_error = np.inf
            #     prev_avg = 0
            #     grad = np.inf
            #     tti_frame_counter = 0
            #     VelocityPrev = np.array([0, 0])

            #     if LSTM or TRANSFORMER:
            #         vel_history  = [[0.0, 0.0]] * HISTORY_LENGTH
            #         acc_history  = [[0.0, 0.0]] * HISTORY_LENGTH
            #         feature_history = [[0.0, 0.0, 0.0, 0.0]] * HISTORY_LENGTH

            #     if KALMANFILTER:
            #         kf_target.x = np.array([[0], [0], [0], [0]])

            #     if LINEARMODEL:
            #         velocity_history = []

            #     print("Reset complete. Ready for next intercept.")

        

        cv.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #print(f"FPS: {fps:.1f} | Frame time: {full_frame_time*1000:.2f} ms")

        cv.imshow("frame", frame)

        # Save video
        if SAVE_VIDEO:
            out.write(frame)
        
        # Exit on 'q' key or ESC: 27
        key = cv.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        # Python FPS
        elapsed_py = time.perf_counter() - fps_python

        if DEBUG:
            print("Python FPS",1/elapsed_py)
            print("AI FPS",1/elapsed_ai)
            print("Contour FPS",1/elapsed_cont)
            print("Cont + AI",1/(elapsed_ai+elapsed_cont))

        # Sleep to hit target FPS
        elapsed = time.perf_counter() - fps_time
        sleep_time = (1 / FPS) - elapsed
        if sleep_time > 0 and REAL_SPEED:
            time.sleep(sleep_time)

        # Measure FPS AFTER sleep, using full frame duration
        full_frame_time = time.perf_counter() - fps_time
        fps_time = time.perf_counter()  # reset for NEXT frame

        #dt = full_frame_time # KF time

        fps = 1 / full_frame_time
        fps_list.append(fps)
   

    if fps_list:
        print(f"\nAverage FPS: {sum(fps_list) / len(fps_list):.1f}")
        print(f"Min FPS: {min(fps_list):.1f}")
        print(f"Max FPS: {max(fps_list):.1f}")

    
    errors = np.array(Errorlist).flatten()
    avgError = np.mean(errors)
    errors_mm = np.array(mean_mm_list).flatten()
    avg_mm = np.mean(errors_mm)
    if errors.size > 0:
        print("Average error for object (mm): ", avg_mm)
        print("Average error for object (pixels): ", avgError)
        print("Mean:", np.mean(errors))
        print("Median:", np.median(errors))
        print("Std:", np.std(errors))
        print("95th percentile:", np.percentile(errors, 95))

    #out.release()
    Vid.release()
    cv.destroyAllWindows()
