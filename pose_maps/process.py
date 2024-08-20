import pyautogui
pyautogui.PAUSE = 0.01
import mediapipe as mp
from typing import NamedTuple
import time

class Pose2MouseMovement:
    def __init__(self) -> None:
        self.screen_width, self.screen_height = pyautogui.size()  # 屏幕参数
        self.mp_hands = mp.solutions.hands  # 手部参数

        # EMA 平滑参数
        self.alpha = 0.2  # 平滑因子, 值越小越平滑

        # 初始化历史坐标
        self.prev_x, self.prev_y = 0, 0

    def classify(self, results: NamedTuple):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
            
                # 获取大拇指的坐标
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                # 获取大拇指第二节坐标
                thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
                # 获取食指顶端坐标
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # 获取中指坐标
                middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                x = thumb_tip.x
                y = thumb_tip.y

                # 将手部坐标转换为屏幕坐标
                screen_x = max(0, min(self.screen_width - int(x * self.screen_width * 4 / 3), self.screen_width))
                screen_y = max(0, min(self.screen_height, int(y * self.screen_height * 4 / 3)))

                # 平滑处理 (EMA)
                smoothed_x = self.alpha * screen_x + (1 - self.alpha) * self.prev_x
                smoothed_y = self.alpha * screen_y + (1 - self.alpha) * self.prev_y

                # 更新历史坐标
                self.prev_x, self.prev_y = smoothed_x, smoothed_y
                
                thumb_index_distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + 
                                        (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
                middle_distance = ((middle_tip.x - thumb_tip.x) ** 2 + 
                                        (middle_tip.y - thumb_tip.y) ** 2) ** 0.5
                threshold_distance = ((thumb_ip.x - thumb_tip.x) ** 2 + 
                                        (thumb_ip.y - thumb_tip.y) ** 2) ** 0.5 / 2
                if thumb_index_distance < threshold_distance and middle_distance > threshold_distance:
                    pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.01)
                    pyautogui.click()
                elif thumb_index_distance > threshold_distance and middle_distance < threshold_distance: 
                    pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.01)
                    pyautogui.rightClick()
                elif thumb_index_distance < threshold_distance and middle_distance < threshold_distance:  
                    pyautogui.dragTo(smoothed_x, smoothed_y, button="left")
                    pyautogui.click()
                else:
                    pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.01)
                break
        return {
            "status": "SUCCESS"
        }
    
    def get_two_finger_distance(self, hand_landmarks, first_finger, second_finger) -> float:
        first = hand_landmarks.landmark[first_finger]
        second = hand_landmarks.landmark[second_finger]
        return ((first.x - second.x) ** 2 + 
                (first.y - second.y) ** 2 + 
                (first.z - second.z) ** 2) ** 0.5