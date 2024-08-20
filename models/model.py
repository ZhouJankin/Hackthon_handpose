import cv2
import mediapipe as mp
from typing import NamedTuple

class Image2HandPose:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def process(self, image: cv2.typing.MatLike) -> NamedTuple:
        results = self.hands.process(image)
        return results
    
    def visualize(self, image: cv2.typing.MatLike, results: NamedTuple) -> None:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Hand Gesture to Replace Mouse', image)