import cv2
import numpy as np

#directions = {forward: 0, left: 1, right: -1}

def stop():
    arr = [0, 0]
    show(arr)

def forward():
    arr = [20, 0]
    show(arr)

def left_turn():
    stop()
    show(arr)
    arr[1] = 1
    show(arr)
    
def right_turn():
    stop()
    show(arr)
    arr[1] = -1
    show(arr)
    
def show(arr)
    print(arr)
    
# ---------------------------
# Camera + Obstacle Detection
# ---------------------------

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacle_detected = False
    for cnt in contours:
        area2 = cv2.convexHull(cnt)
        area = cv2.contourArea(area2)
        if area > 1000:  # adjust threshold for your environment
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Obstacle", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
            obstacle_detected = True

            # Decide steering based on obstacle position
            center_x = x + w//2
            frame_center = frame.shape[1] // 2

            if center_x < frame_center - 50:
                 right_turn()   # obstacle left → turn right
            elif center_x > frame_center + 50:
                 left_turn()    # obstacle right → turn left
            else:
                 stop()        # obstacle in front → stop

    if not obstacle_detected:
        forward()  # clear path → move forward

    cv2.imshow("Obstacle View", frame)
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for white color (tune these values)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    # Mask for white regions
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours of white lines
    line_contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_detected = False
    for cnt in line_contours:
         area2 = cv2.convexHull(cnt)
         area = cv2.contourArea(area2)
         if area > 50:  # adjust threshold
             x, y, w, h = cv2.boundingRect(cnt)
             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
             cv2.putText(frame, "Track Line", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                         0.6, (255, 0, 0), 2)

             line_detected = True

            # Steering logic based on line position
             center_x = x + w//2
             frame_center = frame.shape[1] // 2

             if center_x < frame_center - 50:
                 left_turn()
             elif center_x > frame_center + 50:
                 right_turn()
             else:
                 forward()

    if not line_detected:
        forward()  # on line→ move forward

    cv2.imshow("Line view", frame)
    
    h, s, v = cv2.split(hsv)

    # storing individual bright and shadowed regions
    _, bright_mask = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)
    _, shadow_mask = cv2.threshold(v, 50, 255, cv2.THRESH_BINARY_INV)

    # normalising the view from errors
    v_norm = cv2.equalizeHist(v)
    hsv_normalized = cv2.merge([h, s, v_norm])
    result = cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)

    cv2.imshow("Bright regions", bright_mask)
    cv2.imshow("Shadowed regions", shadow_mask)
    cv2.imshow("Corrected view", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
capture.release()
cv2.destroyAllWindows()
