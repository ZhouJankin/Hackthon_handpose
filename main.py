import cv2
import mediapipe as mp

from models.model import Image2HandPose as I2HModel
from pose_maps.process import Pose2MouseMovement as P2MProcess
import time

def main():
    visualize = True
    image_to_hand_pose = I2HModel()
    pose_to_mouse_move = P2MProcess()
    
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  #
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 将BGR图像转换为RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_pose = image_to_hand_pose.process(image)
        
        if visualize:
            image_to_hand_pose.visualize(frame, hand_pose)
            # 按下'Q'键退出
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        message = pose_to_mouse_move.classify(hand_pose)
    if visualize:
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()