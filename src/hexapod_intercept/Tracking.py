import cv2 as cv
import numpy as np
import math

class Tracking():
    def __init__(self):
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        self.fill = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21,21))
        self.lower_blue = np.array([90, 50, 50]).astype(int)
        self.upper_blue = np.array([130, 255, 255]).astype(int)

        
    def nothing(self, x):
        pass
    
    def Contours(self,frame,filter,DEBUG = False):

        # Scale down for quicker computation
        scale = 0.5
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        frame = cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)

        #frame = cv.GaussianBlur(frame, (3, 3), 0) # 23 = 21
        #frame = cv.medianBlur(frame, 15)  # kernel must be odd
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #frame = self.gray_world(frame)

        
            
        # Filter object for mask, blue/grey
        if filter == 'blue':
            
            #Extract Centers based on blue object
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            h, s, v = cv.split(hsv)
            #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            #clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(24,24))
            #v = clahe.apply(v)
            
            v = cv.normalize(v, None, 0, 255, cv.NORM_MINMAX) # global norm
        
            hsv = cv.merge([h, s, v])

            #lower_blue = np.array([100, 70, 25])
            #upper_blue = np.array([130, 210, 180])
            
            #lower_blue = np.array([90, 60, 20])
            #upper_blue = np.array([130, 255, 255])

            
            mask = cv.inRange(hsv, self.lower_blue, self.upper_blue)



        # if filter == 'blue2':
        #     lab = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        #     l, a, b = cv.split(lab)
        #     clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #     l_eq = clahe.apply(l)
        #     lab_eq = cv.merge([l_eq, a, b])

        #     lower = np.array([0, 70, 0])
        #     upper = np.array([255, 150, 120])

        #     # Apply LAB threshold
        #     mask = cv.inRange(lab_eq, np.array(lower, dtype=np.uint8),
        #                             np.array(upper, dtype=np.uint8))

        # Clean mask
        # Remove small noise
        #opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel,iterations=1)
        
        # Fill small holes
        cleaned_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.fill,iterations=3)






            # lab = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
            # L, A, B = cv.split(lab)
            # lower = np.array([0, 70, 0])
            # upper = np.array([255, 140, 110])
            # mask_raw = cv.inRange(lab, lower, upper)

            # # erode then fill
            # # Clear small noise
            # mask = cv.morphologyEx(mask_raw, cv.MORPH_OPEN, self.kernel)

            # # fill then erode
            # mask = cv.morphologyEx(mask_raw, cv.MORPH_CLOSE, self.fill, iterations=2)
            
            # result = cv.bitwise_and(frame, frame, mask=mask)

            # b, g, r = cv.split(frame.astype(np.float32))
            # total = b + g + r + 1e-6

            # if cv.getWindowProperty('Blue Mask Tuner', cv.WND_PROP_VISIBLE) < 1:
            #     cv.namedWindow('Blue Mask Tuner')
            #     cv.createTrackbar('b_ratio min %', 'Blue Mask Tuner', 0, 100, self.nothing)
            #     cv.createTrackbar('b>g margin %',  'Blue Mask Tuner', 0, 100, self.nothing)
            #     cv.createTrackbar('b>r margin %',  'Blue Mask Tuner',  0, 100, self.nothing)
            #     cv.imshow('Blue Mask Tuner', np.zeros((1, 400, 3), dtype=np.uint8))
            #     cv.waitKey(1)

            

            # # Normalize all RGB values, r + b + g = 1, turns them into %
            # b_ratio = b / total
            # g_ratio = g / total
            # r_ratio = r / total

            # b_min     = cv.getTrackbarPos('b_ratio min %', 'Blue Mask Tuner') / 100.0
            # bg_margin = cv.getTrackbarPos('b>g margin %',  'Blue Mask Tuner') / 100.0
            # br_margin = cv.getTrackbarPos('b>r margin %',  'Blue Mask Tuner') / 100.0

            # blue_mask = (         
            #     (b_ratio > 0.20) &                   # blue > 21%
            #     (b_ratio > g_ratio + 0.04) &        # blue > green + 4%  5-6
            #     (b_ratio > r_ratio + 0.04)          # blue > red + 4%
            # )

           
            
            #mask = (blue_mask * 255).astype(np.uint8)
            #cv.imshow("result",mask)
            #cv.imshow("mask_raw",mask_raw)
            
        #else:
            #Extract Centres based on lighting
            #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            #_, mask = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)

       

        contours, _ = cv.findContours(cleaned_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        centres = []

        #cv.drawContours(cleaned_mask, contours, -1, (0, 255, 0), 2)

        if contours:

            largest_contour = max(contours, key=cv.contourArea)
            Area = cv.contourArea(largest_contour)
            #print("Largest Contour Area (pix^2): ",Area)
            
            if  Area > 100:
                #print(f"contourArea = {cv.contourArea(largest_contour)}")
                M = cv.moments(largest_contour)
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    cx /= scale
                    cy /= scale
                    centres.append((cx, cy))
                
                # cleaned_mask = cv.cvtColor(cleaned_mask, cv.COLOR_GRAY2BGR)
                # rect = cv.minAreaRect(largest_contour)
                # box = cv.boxPoints(rect)
                # box_int = box.astype(int)
                # cv.drawContours(cleaned_mask,[box_int],0,(0,0,255),2)

                # center_x = int(np.mean(box[:, 0]))
                # center_y = int(np.mean(box[:, 1]))
                # center_x /= scale
                # center_y /= scale
                # centres = []
                # centres.append((center_x, center_y))

        if DEBUG:
            return centres,cleaned_mask
        
        else: 
            return centres, None

    def match_centroids(self,prev_centers, curr_centers, max_dist=50):
        """
        prev_centers: [[x,y], [x,y], ...]
        curr_centers: [[x,y], [x,y], ...]
        Returns: ordered list of curr_centers matched to prev_centers
        """
        if len(prev_centers) == 0:
            return curr_centers[:]  # first frame
        matched = []
        used = set()
        for px, py in prev_centers:
            best_idx = None
            best_dist = max_dist

            for idx, (cx, cy) in enumerate(curr_centers):
                if idx in used:
                    continue

                d = math.hypot(cx - px, cy - py)

                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            if best_idx is not None:
                matched.append(curr_centers[best_idx])
                used.add(best_idx)
            else:
                # object disappeared → placeholder or drop it
                matched.append([px, py])

        return matched

    def ErrorTracking(self,centers,Predict,PredictFrame):
        centers = np.array(centers)
        Predict = np.array(Predict)

        error = np.linalg.norm(centers - Predict, axis=1)
        mean_error = np.mean(error)
        print(mean_error)
        
        return mean_error
    

    def ResizeIMG(self,frame):
        h, w = frame.shape[:2]
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)

        coords = cv.findNonZero(mask)
        if coords is None:
            return frame

        x, y, w, h = cv.boundingRect(coords)

        if h > w:
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

        return frame[x:x+w, y:y+h]
    
    def order_points(self, pts):
        # pts is a (4, 2) array of coordinates
        rect = np.zeros((4, 2), dtype="float32")

        # Top-left has the smallest sum, bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-Left
        rect[2] = pts[np.argmax(s)] # Bottom-Right

        # Compute the difference between the points
        # Top-right has the smallest difference, bottom-left has the largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-Right
        rect[3] = pts[np.argmax(diff)] # Bottom-Left

        return rect
    
    def gray_world(self, frame):
        """Apply gray‑world white balance to normalise left/right colour cast."""
        b, g, r = cv.split(frame.astype(np.float32))
        # Average value of each channel
        b_avg, g_avg, r_avg = cv.mean(b)[0], cv.mean(g)[0], cv.mean(r)[0]
        # Overall gray value (usually green channel weight)
        gray_val = (b_avg + g_avg + r_avg) / 3.0
        # Scale factors
        kb = gray_val / (b_avg + 1e-6)
        kg = gray_val / (g_avg + 1e-6)
        kr = gray_val / (r_avg + 1e-6)
        # Apply
        b = np.clip(b * kb, 0, 255).astype(np.uint8)
        g = np.clip(g * kg, 0, 255).astype(np.uint8)
        r = np.clip(r * kr, 0, 255).astype(np.uint8)
        return cv.merge([b, g, r])
    
    def Crop_Arena(self,frame,TARGET_W,TARGET_H):
        x = None
        # HSV used for better color segmentation
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        l_s = 45 # Lower saturation
        l_v = 55 # Lower value (brightness)
        lower_red1 = np.array([0, l_s, l_v], dtype=np.uint8)
        upper_red1 = np.array([15, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([165, l_s, l_v], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        # 2 masks for red hue range (0-10 and 170-180)
        mask1 = cv.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        # Find contours in the red mask to locate the arena
        contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Process contours to find the largest red area which = arena
        # Aproximatly get the corners of the areana
        for cont in contours:
            # Use convex hull to get a smoother contour
            rect = cv.convexHull(cont)
            # Calculate the perimeter to find epsilon
            peri = cv.arcLength(rect, True)
            # Approximate the polygon to 4 points
            approx = cv.approxPolyDP(rect, 0.02 * peri, True)
        
        # Filter out small perimeters
            if peri > 1000:
                # Need to redifine wihout points
                hull = cv.convexHull(cont,returnPoints = False)
                # Weird error "The convex hull indices are not monotonous"
                hull = np.sort(hull, axis=0)
                # Find convexity defects to get the rough points of the contour
                # Gets the aproximate points areana tape corners
                defects = cv.convexityDefects(cont,hull)

                if defects is None:
                    continue

                pointlist2 = []
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    far = tuple(cont[f][0])
                    pointlist2.append(far)

                pointlist2 = np.array(pointlist2, dtype=np.int32)
                x,y,w,h = cv.boundingRect(pointlist2)
                
        if x != None:
            Corner_pts = np.array([[x, y],[x + w, y],[x + w, y + h],[x, y + h]]).astype(np.float32)
            Original_pts = np.array([[0,0],[TARGET_W,0],[TARGET_W,TARGET_H],[0,TARGET_H]]).astype(np.float32)

            # Warp to rough area of arena, to filter donw more
            M1 = cv.getPerspectiveTransform(Corner_pts, Original_pts)
            crop_img = cv.warpPerspective(frame, M1, (TARGET_W, TARGET_H), flags=cv.INTER_CUBIC)

            # Redo red mask on cropped image
            crop_hsv = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)
            mask1 = cv.inRange(crop_hsv, lower_red1, upper_red1)
            mask2 = cv.inRange(crop_hsv, lower_red2, upper_red2)
            red_mask2 = mask1 | mask2

            # Get the inverse to get the area without the red tape, which is the actual arena
            Background_mask = cv.bitwise_not(red_mask2)

            # cv.RETR_LIST gives all contours without hierarchy
            contours, _ = cv.findContours(Background_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            
            for cont in contours:
                area = cv.contourArea(cont)
                
                if area > 210000: #136895.0 area of biggest box
                    
                    #print("Found area: ",area)

                    # Use convex hull to get a smoother contour
                    rect = cv.convexHull(cont)
                    # Calculate the perimeter to help define epsilon
                    peri = cv.arcLength(rect, True)
                    # Approximate the polygon to 4 points
                    approx = cv.approxPolyDP(rect, 0.02 * peri, True)
                    cv.drawContours(crop_img, [approx], 0, (255, 0, 0), 3)
                else:
                    continue
                    print("None found")
            
            if len(approx) == 4:
                Arena_pts = np.array(approx).reshape(-1, 2).astype(np.float32)
                Arena_pts = self.order_points(Arena_pts)
                
                # Trim border for less areana 
                BORDER_TRIM = -5
                Crop_pts = np.array([
                    [BORDER_TRIM,            BORDER_TRIM],
                    [TARGET_W - BORDER_TRIM, BORDER_TRIM],
                    [TARGET_W - BORDER_TRIM, TARGET_H - BORDER_TRIM],
                    [BORDER_TRIM,            TARGET_H - BORDER_TRIM],
                ], dtype = np.float32)

                # Warp to get the final cropped image of the arena
                M2 = cv.getPerspectiveTransform(Arena_pts, Crop_pts)
                #Arena = cv.warpPerspective(crop_img, M2, (TARGET_W, TARGET_H),flags =cv.INTER_NEAREST)
            
                # Debug
                # cv.imshow("Background Mask",Background_mask)
                # cv.imshow("crop", crop_img)
                # cv.waitKey(0)
                
                
                # Return the matrix multiplication from both frames 
                return M2 @ M1
        
        # Debug
        # cv.imshow("Background Mask",Background_mask)
        # cv.imshow("crop", crop_img)
        # cv.waitKey(0)
        
        return None
    


