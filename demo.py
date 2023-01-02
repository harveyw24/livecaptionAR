import cv2
import datetime

vid_capture = cv2.VideoCapture(0)

vid_cod = cv2.VideoWriter_fourcc(*'MJPG')

name = '_'.join(['orig_vid', datetime.datetime.now().strftime("%y%m%d_%H%M%S")])
output = cv2.VideoWriter(f'data/orig_vids/{name}.mp4', vid_cod, 25.0, (854, 480))

while(True):
    # Capture each frame
    ret, frame = vid_capture.read()
    cv2.imshow("Recording in progress...", frame)
    output.write(frame)

    if cv2.waitKey(1) &0XFF == ord(' '):
        break

vid_capture.release()
output.release()
cv2.destroyAllWindows()