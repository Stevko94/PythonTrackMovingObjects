import cv2
import time

# Initialize the DroidCam camera
camera = cv2.VideoCapture("http://192.168.0.161:4747/video")

# Read the first frame
ret, frame = camera.read()

# Convert the frame to grayscale for processing
previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Counter for captured images
image_counter = 0

# Path to the folder to save the captured images
output_folder = "output_folder"

# Maximum number of images to capture
max_images = 5

# Minimum contour area for an object
min_contour_area = 500

# Timer variables
start_time = time.time()
elapsed_time = 0

while True:
    # Read the current frame from DroidCam
    ret, frame = camera.read()

    # Convert the frame to grayscale for processing
    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(current_frame, previous_frame)

    # Apply a threshold to the frame difference
    _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Apply image dilation to fill in gaps in the thresholded image
    dilated = cv2.dilate(threshold, None, iterations=2)

    # Find contours of the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding rectangles around the moving objects and capture images of large contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      
      # Save the captured image to the output folde  
        if image_counter < max_images:
              image_path = f"{output_folder}/captured_image_{image_counter}.jpg"
              cv2.imwrite(image_path, frame)
              image_counter += 1

    # Update the previous frame
    previous_frame = current_frame

    # Display the current frame
    cv2.imshow("DroidCam", frame)
    
    # Check if the maximum number of images per minute is reached
    if image_counter >= max_images:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            # Reset the timer and counters
            start_time = time.time()
            image_counter = 0

    # Exit the loop if 'q' is pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
