import cv2
import numpy as np

def show_color_change_window():
    window_name = 'Color Changer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 200, 200)

    red = (0, 0, 255)
    black = (0, 0, 0)
    color = red
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        # Create an image with the current color
        img = np.zeros((200, 200, 3), np.uint8)
        img[:] = color

        if color == red:
            # Put text "Detected" in the center of the image
            textsize = cv2.getTextSize("Detected", font, 1, 2)[0]
            textX = (img.shape[1] - textsize[0]) // 2
            textY = (img.shape[0] + textsize[1]) // 2
            cv2.putText(img, "Detected", (textX, textY), font, 1, black, 2)

        # Display the image
        cv2.imshow(window_name, img)

        # Wait for 2 seconds
        key = cv2.waitKey(1)

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

        # Change color
        color = black if color == red else red

    cv2.destroyAllWindows()

show_color_change_window()

