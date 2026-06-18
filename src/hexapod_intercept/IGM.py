import numpy as np

def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])

A     = np.array([0, -0.24355, -0.21320, 0,       0,      0     ])
ALPHA = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
D     = np.array([0.15185, 0, 0, 0.13105, 0.08535,   0.09210 + 0.175]) # 0.175 is the gripper
THETA = np.zeros(6)

def forward_kin(q):
    T = np.eye(4)
    T_frames = []

    for i in range(6):
        frame = dh_matrix(A[i], ALPHA[i], D[i], THETA[i] + q[i])
        T = T @ frame
        T_frames.append(frame)
    
    return T, T_frames # returns T06 and a list of each frame


def igm(U):
    q = np.zeros(6) # start position
    T, T_frames = forward_kin(q)

    T01, T12, T23, T34, T45, T56 = T_frames

    T56_inv = np.linalg.inv(T56)
    T45_inv = np.linalg.inv(T45)
    T01_inv = np.linalg.inv(T01)

    U15 = T01_inv @ U @ T56_inv
    T15 = T12 @ T23 @ T34 @ T45

    T14 = T01_inv @ U @ np.linalg.inv(T45 @ T56)

    #q1_lhs = 
    #q1_rhs = T15[2,3]

    
    # px * sin(q1) - py * cos(q1) - 0.0921 * ax * sin(q1) + 0.0921 * ay * cos(q1) = 0.13105
    # X = px + (-0.0921 * ax)
    # Y = -py + (0.0921 * ay)

    #   [sx, nx, ax, Px],
    #   [sy, ny, ay, Py],
    #   [sz, nz, az, Pz],
    #   [0, 0, 0, 1]
    
    # 0.1*sin(q1) - 0.1921*cos(q1) = 0.13105
    ax = U[0,2]
    ay = U[1,2]
    az = U[2,2]

    px = U[0,3]
    py = U[1,3]
    pz = U[2,3]

    X = px + (-D[5] * ax)
    Y = -py + (D[5] * ay)
    Z = 0.13105

    #if X**2 + Y**2 <= Z**2:
        #print("No soluation at q1")

    S_q1_1 = (((X * Z) + (Y * np.sqrt(X**2 + Y**2 - Z**2)))/(X**2 + Y**2))
    C_q1_1 = (((Y * Z) - (X * np.sqrt(X**2 + Y**2 - Z**2)))/(X**2 + Y**2))
    q1_sol1 = np.arctan2(S_q1_1,C_q1_1)

    S_q1_2 = (((X * Z) - (Y * np.sqrt(X**2 + Y**2 - Z**2)))/(X**2 + Y**2))
    C_q1_2 = (((Y * Z) + (X * np.sqrt(X**2 + Y**2 - Z**2)))/(X**2 + Y**2))
    q1_sol2 = np.arctan2(S_q1_2,C_q1_2)

    q1_list = [q1_sol1, q1_sol2]

    # ax * sin(q1) + ay * cos(q1) = cos(q5)
    C_q5 = ax * S_q1_1 - ay * C_q1_1
    q5_sol1 =  np.arccos(C_q5)
    q5_sol2 = -q5_sol1

    C_q5 = ax * S_q1_2 - ay * C_q1_2
    q5_sol3 =  np.arccos(C_q5)
    q5_sol4 = -q5_sol3

    q5_list = [q5_sol1,q5_sol2,q5_sol3,q5_sol4]

    # nx*sin(q1) - ny*cos(q1))*cos(q6) + (sx*sin(q1) - sy*cos(q1))*sin(q6) = 0

    # nx*sin(q1) - ny*cos(q1))*cos(q6) = - (sx*sin(q1) - sy*cos(q1))*sin(q6)
    # nx*sin(q1) - ny*cos(q1))/- (sx*sin(q1) - sy*cos(q1)) = sin(q6)/cos(q6)
    # q6
    nx = U[0,1]
    ny = U[1,1]
    nz = U[2,1]

    sx = U[0,0]
    sy = U[1,0]
    sz = U[2,0]

    q6_list = []
    for q1 in q1_list:
        s_q6 = (-nx*np.sin(q1) + ny*np.cos(q1))
        c_q6 = (sx*np.sin(q1) - sy*np.cos(q1))
        q6_1 = np.arctan2(s_q6,c_q6)
        q6_2 = q6_1 + np.pi  # second solution
        q6_list.append(q6_1)
        q6_list.append(q6_2)
        ##print(q6_1, q6_2)
        
    solutions = []

    for i, q1 in enumerate(q1_list):
        start_idx = i * 2
        end_idx   = start_idx + 2
        

        for j in range(start_idx, end_idx):
            q5 = q5_list[j]
            q6 = q6_list[j]
            
            # Rebuild with q values subbed in
            T01 = dh_matrix(A[0], ALPHA[0], D[0], q1)
            T45 = dh_matrix(A[4], ALPHA[4], D[4], q5)
            T56 = dh_matrix(A[5], ALPHA[5], D[5], q6)
            T14 = np.linalg.inv(T01) @ U @ np.linalg.inv(T45 @ T56)

            Z1 = T14[0, 3]
            Z2 = T14[1, 3]
            X  = A[1]   # -0.24355
            Y  = A[2]   # -0.21320

            C_q3 = (Z1**2 + Z2**2 - X**2 - Y**2) / (2*X*Y)

            # Clamp to prevent sqrt of negative
            #C_q3 = np.clip(C_q3, -1.0, 1.0) 
            #if abs(C_q3) > 0.99:
                #print(f"Warning: near workspace boundary, C_q3 = {C_q3:.6f}, solutions may be inaccurate")

            if abs(C_q3) > 1:
                #print("Out of reach")
                continue

            q234 = np.arctan2(T14[1, 0], T14[0, 0])

            signs = [1, -1]
            for sol in signs:
                S_q3 = sol * np.sqrt(1 - C_q3**2)
                q3 = np.arctan2(S_q3, C_q3)

                B1 = X + Y*np.cos(q3)
                B2 = Y*np.sin(q3)
                q2 = np.arctan2(B1*Z2 - B2*Z1, B1*Z1 + B2*Z2)

                q4 = np.arctan2(np.sin(q234 - q2 - q3), np.cos(q234 - q2 - q3)) # wrap around

                solutions.append((q1, q2, q3, q4, q5, q6))

            
    safe_solutions = []
    table_height = -0.010
    for i, sol in enumerate(solutions):
        T_check, frames = forward_kin(sol)

        #error = np.max(np.abs(T_check - U))
        
        #deg = [f"{np.degrees(q):.1f}" for q in sol]
        #print(f"Sol {i+1} | Error: {error}")
        #print(f"   q: {deg}")
        
        # frames[2] is T03 (Elbow), frames[4] is T05 (Wrist)
        z_elbow = frames[2][2, 3]
        z_wrist = frames[4][2, 3]
        
        is_safe = z_elbow >= table_height and z_wrist >= table_height
        
        
        if is_safe: 
            safe_solutions.append(sol)


        # status = "SAFE" if is_safe else "UNDER TABLE"
        #deg = [f"{np.degrees(q):.1f}" for q in sol]
        #print(f"Sol {i+1}: z_elb={z_elbow:+.3f} | {status} | Error: {error}")
        #print(f"   q: {deg}")

    return safe_solutions


def filter_collisions(solutions, table_height=0.0):
    valid_solutions = []
    
    for sol in solutions:
        # Get all transformation matrices for this specific solution
        _, T_frames = forward_kin(sol)
        
        is_safe = True
        # Check the Z-coordinate of Joint 3, 4, and 5
        # T_frames[i] is the matrix for joint i+1. 
        # The position is in the 4th column (index 3), 3rd row (index 2).
        for i in [2, 3, 4]: 
            joint_z = T_frames[i][2, 3]
            if joint_z < table_height:
                is_safe = False
                break
        
        if is_safe:
            valid_solutions.append(sol)
            
    return valid_solutions

def vec2rot(u):
    # Init 4x4 Identity Matrix
    T = np.eye(4)

    px,py,pz,ux,uy,uz = u
    theta = (ux**2 + uy**2 + uz**2)**0.5 # magnitude

    pos = np.array([px,py,pz])

    # Position col
    T[:3, 3] = pos

    # Calculate Rotation
    ux /= theta
    uy /= theta
    uz /= theta
    C = np.cos(theta)
    S = np.sin(theta)
    V = 1 - C

    R = np.array([
        [ux*ux*V + C,    ux*uy*V - uz*S,  ux*uz*V + uy*S],
        [ux*uy*V + uz*S, uy*uy*V + C,     uy*uz*V - ux*S],
        [ux*uz*V - uy*S, uy*uz*V + ux*S,  uz*uz*V + C   ]
    ])

    # Add rotaiotn to homogenous matrix
    T[:3, :3] = R

    return T

def IGM_Fastest_sol(u, Rest_q):
    # Robot dynamics
    q_max       = (32/180) * np.pi # rad/s
    accel       = (100/180) * np.pi     

    r = vec2rot(u)
    all_solutions = igm(r) # returns list of 6‑element tuples
    if not all_solutions:
        return None

    solutions = np.array(all_solutions) # shape (n, 6)
    rest = np.array(Rest_q)

    # precomputed constants
    t_acc   = q_max / accel
    d_acc   = 0.5 * accel * t_acc**2

    best_time = float('inf')
    best_q    = None

    for sol in solutions:
        # shortest angular distance wrapped
        diffs = (sol - rest + np.pi) % (2 * np.pi) - np.pi
        distances = np.abs(diffs)

        # compute time for each joint, take the maximum
        max_joint_time = 0.0
        for dist in distances:
            if dist >= 2 * d_acc:
                # Trapezoidal, ramp up then cruise then ramp down
                t_cruise = (dist - 2 * d_acc) / q_max
                t_total  = 2 * t_acc + t_cruise
            else:
                # Triangular, never reaches q_max
                v_peak   = np.sqrt(accel * dist)
                t_total  = 2 * v_peak / accel
            if t_total > max_joint_time:
                max_joint_time = t_total

        if max_joint_time < best_time:
            best_time = max_joint_time
            best_q    = sol

    return best_q





