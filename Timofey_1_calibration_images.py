import cv2
from pynput import keyboard
import matplotlib.pyplot as plt

# Use direct show flag if on Windows: cv2.CAP_DSHOW
cap = cv2.VideoCapture(0)

# Define resolution of your web-camera
width = 1920
height = 1080

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

scale = 2  # Scale image to visualize
num = 0  # Counter for calibration images
save = False # Image save flag
terminate = False # Appicatiion termniation flag


def on_key_press(key):
    global save, terminate
    if key == keyboard.Key.shift:
        print("Image saved!")
        save = True
    if key == keyboard.Key.esc:
        print("###### TERMINATE APPLICATION ######")
        terminate = True


listener = keyboard.Listener(on_press=on_key_press)
listener.start()

while cap.isOpened():

    succes, img = cap.read()

    cv2.imshow('Img', cv2.resize(
        img, (img.shape[1]//scale, img.shape[0]//scale)))
    cv2.waitKey(1)

    
    if save:
        # Save images (.png, 300 dpi) in calib_images folder using plt.imsave
        plt.imsave(f'./data/calibrations/' + str(num) + '.png',
                   cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dpi=300)
        num += 1
        save = not save
    
    if terminate:
        break

cap.release()
cv2.destroyAllWindows()
