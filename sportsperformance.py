import cv2
import numpy as np
cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()
heatmap = None
trajectory = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 360))
    fg_mask = backSub.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_positions = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w // 2, y + h // 2)
            player_positions.append(center)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)
            trajectory.append(center)
    '''for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 1)'''
    if heatmap is None:
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), np.float32)
    for pos in player_positions:
        cv2.circle(heatmap, pos, 15, 1, -1)
    heatmap_img = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_img = np.uint8(heatmap_img)
    heatmap_img_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, heatmap_img_color, 0.3, 0)
    cv2.imshow('Player Tracking and Heatmap', overlay)
    if cv2.waitKey(30) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()