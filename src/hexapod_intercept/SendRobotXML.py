import urlib
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import numpy as np
import threading
import time  

YRES = 720 
XRES = 1280
X,Y = None,None
PX,PY = None,None
Target_angle = None
SPEED = None


# class QuietRequestHandler(SimpleXMLRPCRequestHandler):
#     def log_message(self, format, *args):
#         return  # Disable POST request logging


point_ready = threading.Event()
def UpdatePoints(x, y):
    global X, Y, _last_point_sent_time
    X = x.item()
    Y = y.item()
    _last_point_sent_time = time.perf_counter() 
    point_ready.set()

def wait_point():
    point_ready.wait()
    point_ready.clear()
    return True

def PixelToPoseX(pixel):
    x_start = -0.010812152843194799
    x_end   = 0.46

    return x_start + (pixel / (XRES - 1)) * (x_end - x_start)

def PixelToPoseY(pixel):
    y_start = -0.19926
    y_end   = -0.32166

    return y_start + (pixel / (YRES - 1)) * (y_end - y_start)

def get_next_pose(Original_pose):

        #print(f"Converting to metres x = {X}, y = {Y}")

        # Convert X and Y into metres
        x = PixelToPoseX(X)
        y = PixelToPoseY(Y)

        # Put into a list for ease of reading and editing, only put x and y as the robot is perpindicular to the areana
        # Test out only having Z defined
        #       x  y     z    rx     ry       rz
        pose = [x, y, 0.100, 2.28, -2.161, 0] 
        print("Pose: ", pose)
        #pose = [x, y, 0.00, -0.125, 0.005, -1.513]    # z = -0.005 for almost touching table
        #print(f"Sent x = {x}, y = {y}")

        if X == None or Y == None:
            return Original_pose
        
        return urlib.listToPose(pose)


angle_ready = threading.Event()

def send_angle(angles): # UR robot sends angles of all its joints, Used for calculating time to prediction
    global Target_angle
    Target_angle = angles
    angle_ready.set()
    #print("Recieved angle:",Target_angle)

def get_angle():
    global Target_angle
    
    angle_ready.wait() # Wait for robot to send cooridinated
    angle_ready.clear()
    
    angle = Target_angle
    Target_angle = None
    return angle
         
connected = threading.Event()
def handshake():
    print("Client connected")
    connected.set()
    return True


_last_point_sent_time = 0.0
_latest_latency_sec   = 0.0      # most recent measured latency
_latency_lock = threading.Lock()  # thread‑safe access

def calibrate_latency(): # UR3e calls this to start timing latency
    global _last_point_sent_time
    with _latency_lock:
        _last_point_sent_time = time.perf_counter()
    return True

robot_stopped = threading.Event()
def set_robot_moving(moving):
    global _last_point_sent_time, _latest_latency_sec
    if moving == False:
        robot_stopped.set()
        return False
    
    # elif moving == True:
    #     robot_stopped.clear()
    #     return True
    
    
    elif moving == True:
        robot_stopped.clear()
        # measure command latency
        with _latency_lock:
            if _last_point_sent_time > 0:
                now = time.perf_counter()
                _latest_latency_sec = now - _last_point_sent_time
                print(f"Measured command latency: {_latest_latency_sec*1000:.1f} ms")
                _last_point_sent_time = 0.0
        return True

    else:
        print("Error neither True or False")
     
def is_robot_stopped():
    return robot_stopped.is_set() #return true if the internal flag is true

def get_latency():
    # Return the most recently measured command latency in seconds
    with _latency_lock:
        return _latest_latency_sec
    
def XMLServer():
    #server = SimpleXMLRPCServer(("", 50000), allow_none=True, requestHandler=QuietRequestHandler) # Request handler turns off print
    server = SimpleXMLRPCServer(("", 50000), allow_none=True)
    server.RequestHandlerClass.protocol_version = "HTTP/1.0"
    print("Listening on port 50000...")
    server.register_function(handshake, "handshake") # Wait for UR to connect to pc
    server.register_function(get_next_pose, "get_next_pose") #input current pos to get next pos in milimetres
    server.register_function(set_robot_moving, "set_robot_moving") # handshake for when robot is moving
    server.register_function(send_angle, "send_angle") # Retrieve angle from UR robot
    server.register_function(wait_point, "wait_point") # Retrieve angle from UR robot
    server.register_function(calibrate_latency, "calibrate_latency")

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    print("Waiting for robot connection...")
    connected.wait()
    print("Robot connected")




"""
Listening on port 50000...
UR Message:  {'rx': 0.03285359370200006, 'ry': -0.16586596196300035, 'rz': -1.425509397299, 'x': 0.46638700735900135, 'y': -0.3459628186910013, 'z': -0.005117079554000203}
192.168.3.2 - - [25/Nov/2025 12:40:34] "POST / HTTP/1.1" 200 -
UR Message:  {'rx': -0.12562496933545336, 'ry': 0.005181453263110702, 'rz': -1.51388935589199, 'x': -0.010812152843194799, 'y': -0.3309808481437849, 'z': -0.00530222776668543}
192.168.3.2 - - [25/Nov/2025 12:40:37] "POST / HTTP/1.1" 200 - 

-----------------------------------------
start x = 0.46638700735900135 metres
end x = -0.010812152843194799 metres
dist = 0.477199160202196149 metres
-----------------------------------------
start y = -0.3412 metres
end y = -0.18343 metres
displacement = 0.12238 metres
-----------------------------------------

0.4484640342058672



"""