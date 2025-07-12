import cv2
import numpy as np

# Initialize camera
camera = cv2.VideoCapture(0)

# Create a canvas to draw on
canvas = None

# Previous coordinates
prev_x, prev_y = None, None
smooth_x, smooth_y = None, None
alpha = 0.2  # Smoothing factor

# Create trackbars for HSV thresholding
cv2.namedWindow("Trackbars")
cv2.createTrackbar('Lower Hue', 'Trackbars', 0, 179, lambda x: None)
cv2.createTrackbar('Upper Hue', 'Trackbars', 179, 179, lambda x: None)
cv2.createTrackbar('Lower Saturation', 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar('Upper Saturation', 'Trackbars', 255, 255, lambda x: None)
cv2.createTrackbar('Lower Value', 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar('Upper Value', 'Trackbars', 255, 255, lambda x: None)

def get_color_name(hue, saturation, value):
    if value < 50:
        return "Black"
    elif saturation < 50 and value > 200:
        return "White"
    elif saturation < 50:
        return "Gray"
    elif hue < 10 or hue > 160:
        return "Red"
    elif 10 < hue <= 25:
        return "Orange"
    elif 25 < hue <= 35:
        return "Yellow"
    elif 35 < hue <= 85:
        return "Green"
    elif 85 < hue <= 125:
        return "Blue"
    elif 125 < hue <= 160:
        return "Purple"
    else:
        return "Unknown"

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hue = cv2.getTrackbarPos('Lower Hue', 'Trackbars')
    upper_hue = cv2.getTrackbarPos('Upper Hue', 'Trackbars')
    lower_saturation = cv2.getTrackbarPos('Lower Saturation', 'Trackbars')
    upper_saturation = cv2.getTrackbarPos('Upper Saturation', 'Trackbars')
    lower_value = cv2.getTrackbarPos('Lower Value', 'Trackbars')
    upper_value = cv2.getTrackbarPos('Upper Value', 'Trackbars')

    lower_bound = np.array([lower_hue, lower_saturation, lower_value])
    upper_bound = np.array([upper_hue, upper_saturation, upper_value])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours of the object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if center:
                if smooth_x is None or smooth_y is None:
                    smooth_x, smooth_y = center
                else:
                    smooth_x = int((1 - alpha) * smooth_x + alpha * center[0])
                    smooth_y = int((1 - alpha) * smooth_y + alpha * center[1])

                smoothed_center = (smooth_x, smooth_y)

                # Draw dot
                #cv2.circle(frame, smoothed_center, 7, (0, 0, 255), -1)

                # Display detected color name
                h, s, v = hsv[smoothed_center[1], smoothed_center[0]]
                color_text = get_color_name(h, s, v)
                cv2.putText(frame, color_text, (smoothed_center[0] - 20, smoothed_center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw if movement is significant
                if prev_x is not None and prev_y is not None:
                    distance = np.hypot(smooth_x - prev_x, smooth_y - prev_y)
                    if distance > 5:
                        cv2.line(canvas, (prev_x, prev_y), smoothed_center, (0, 0, 255), 5)

                prev_x, prev_y = smoothed_center
        else:
            prev_x, prev_y = None, None
            smooth_x, smooth_y = None, None
    else:
        prev_x, prev_y = None, None
        smooth_x, smooth_y = None, None

    # Combine canvas with frame
    frame = cv2.add(frame, canvas)

    # Show the output windows
    cv2.imshow("Virtual Paint", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros_like(frame)

camera.release()
cv2.destroyAllWindows()
