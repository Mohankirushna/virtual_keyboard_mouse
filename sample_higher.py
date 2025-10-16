import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

class VirtualKeyboardMouse:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Mouse control variables
        self.prev_x, self.prev_y = 0, 0
        self.smoothening = 7
        self.click_threshold = 30
        self.last_click_time = 0
        self.click_cooldown = 0.5
        
        # Keyboard layout
        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.'],
            ['SPACE', 'BACKSPACE', 'ENTER']
        ]
        
        self.mode = "MOUSE"  # MOUSE or KEYBOARD
        self.typed_text = ""
        
    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def draw_keyboard(self, img):
        """Draw transparent, centered virtual keyboard on screen with improved UI"""
        h, w = img.shape[:2]
        key_w, key_h = 70, 70
        margin = 15
        start_y = (h - ((len(self.keys) * (key_h + margin)) + 50)) // 2

        # Create overlay for transparency
        overlay = img.copy()
        alpha = 0.5  # Transparency factor

        button_list = []
        
        for i, row in enumerate(self.keys):
            row_width = sum([key_w * (3 if key == "SPACE" else 2 if key in ["BACKSPACE", "ENTER"] else 1) + margin for key in row]) - margin
            start_x = (w - row_width) // 2

            x = start_x
            for key in row:
                if key == "SPACE":
                    w_key = key_w * 3
                elif key in ["BACKSPACE", "ENTER"]:
                    w_key = key_w * 2
                else:
                    w_key = key_w

                # Draw rounded rectangle (modern look)
                cv2.rectangle(overlay, (x, start_y), (x + w_key, start_y + key_h), (50, 50, 50), -1, cv2.LINE_AA)
                cv2.rectangle(overlay, (x, start_y), (x + w_key, start_y + key_h), (255, 255, 255), 2, cv2.LINE_AA)

                font_scale = 0.7 if len(key) > 1 else 1.0
                text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                text_x = x + (w_key - text_size[0]) // 2
                text_y = start_y + (key_h + text_size[1]) // 2

                cv2.putText(overlay, key, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

                button_list.append([key, x, start_y, w_key, key_h])
                x += w_key + margin

            start_y += key_h + margin

        # Display typed text area (transparent)
        cv2.rectangle(overlay, (50, 50), (w - 50, 130), (40, 40, 40), -1, cv2.LINE_AA)
        cv2.putText(overlay, self.typed_text[-30:], (70, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Blend overlay with original image
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        return button_list
    def check_key_press(self, finger_tip, button_list):
        """Check if finger is pressing a key"""
        for key, x, y, w, h in button_list:
            if x < finger_tip[0] < x + w and y < finger_tip[1] < y + h:
                return key
        return None
    def is_left_hand(self, hand_landmarks):
      """Returns True if detected hand is left"""
      handedness = hand_landmarks.classification[0].label if hasattr(hand_landmarks, 'classification') else None
      return handedness == "Left"

    def is_fist(self, lm_list):
        """Returns True if all fingers are folded (fist gesture)"""
        # Check if tips of fingers are below their respective pip joints
        # Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
        # PIP joints: 3, 6, 10, 14, 18
        if len(lm_list) < 21:
            return False
        folded = 0
        for tip, pip in zip([8,12,16,20], [6,10,14,18]):
            if lm_list[tip][1] > lm_list[pip][1]:  # y increases downward
                folded += 1
        # Thumb: check if tip is close to palm
        thumb_folded = abs(lm_list[4][0] - lm_list[2][0]) < 40
        return folded == 4 and thumb_folded

    def run(self):
      cap = cv2.VideoCapture(0)
      cap.set(3, 1280)
      cap.set(4, 720)
        
      frame_count = 0
      fps_time = time.time()
      fps = 0
        
      print("Controls:")
      print("- Press 'K' to switch to Keyboard mode")
      print("- Press 'M' to switch to Mouse mode")
      print("- Press 'Q' to quit")
      print("\nMouse Mode:")
      print("- Move index finger to control cursor")
      print("- Pinch index and thumb to click")
      print("\nKeyboard Mode:")
      print("- Point at keys to type")
      print("- Pinch index and thumb to press key")
      print("TIP: Click/select the search box in your browser before typing with the virtual keyboard.")
        
      while True:
        success, img = cap.read()
        if not success:
          break
            
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)
            
        # Calculate FPS
        frame_count += 1
        if frame_count >= 10:
          fps = 10 / (time.time() - fps_time)
          fps_time = time.time()
          frame_count = 0
            
        # Draw mode indicator
        mode_color = (0, 255, 0) if self.mode == "MOUSE" else (0, 165, 255)
        cv2.putText(img, f"Mode: {self.mode}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        cv2.putText(img, f"FPS: {int(fps)}", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Always define button_list
        button_list = None
        if self.mode == "KEYBOARD":
          button_list = self.draw_keyboard(img)
            
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(img, hand_landmarks, 
                                               self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get landmarks
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
              lm_list.append([int(lm.x * w), int(lm.y * h)])
                    
                    # Detect left hand fist to toggle mode
                    # Use handedness from results if available
              handedness = None
              if hasattr(results, 'multi_handedness') and results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
                is_left = handedness == "Left"
              if is_left and self.is_fist(lm_list):
                current_time = time.time()
                if current_time - self.last_click_time > self.click_cooldown:
                  self.mode = "KEYBOARD" if self.mode == "MOUSE" else "MOUSE"
                  self.last_click_time = current_time
                    
              if len(lm_list) >= 21:
                # Index finger tip and thumb tip
                index_tip = lm_list[8]
                thumb_tip = lm_list[4]
                        
                        # Draw circles on fingertips
                cv2.circle(img, tuple(index_tip), 10, (0, 255, 255), -1)
                cv2.circle(img, tuple(thumb_tip), 10, (255, 0, 255), -1)
                        
                        # Calculate distance between thumb and index
                dist = self.distance(index_tip, thumb_tip)
                if self.mode == "MOUSE":
                            # Mouse control
                            # Map camera coordinates to screen
                  screen_x = np.interp(index_tip[0], [100, w-100], [0, self.screen_w])
                  screen_y = np.interp(index_tip[1], [100, h-100], [0, self.screen_h])
                            
                            # Smooth cursor movement
                  curr_x = self.prev_x + (screen_x - self.prev_x) / self.smoothening
                  curr_y = self.prev_y + (screen_y - self.prev_y) / self.smoothening
                            
                  pyautogui.moveTo(curr_x, curr_y)
                  self.prev_x, self.prev_y = curr_x, curr_y
                            
                            # Click detection (pinch gesture)
                  if dist < self.click_threshold:
                    cv2.circle(img, tuple(index_tip), 15, (0, 255, 0), -1)
                    current_time = time.time()
                    if current_time - self.last_click_time > self.click_cooldown:
                      pyautogui.click()
                      self.last_click_time = current_time
                elif self.mode == "KEYBOARD":
                  # Only call check_key_press if button_list is not None
                  if button_list is not None:
                    pressed_key = self.check_key_press(index_tip, button_list)
                            
                    if pressed_key and dist < self.click_threshold:
                      current_time = time.time()
                      if current_time - self.last_click_time > self.click_cooldown:
                                    # Send real keystrokes to system
                        if pressed_key == "SPACE":
                          pyautogui.press('space')
                          self.typed_text += " "
                        elif pressed_key == "BACKSPACE":
                          pyautogui.press('backspace')
                          self.typed_text = self.typed_text[:-1]
                        elif pressed_key == "ENTER":
                          pyautogui.press('enter')
                          self.typed_text += "\n"
                        else:
                          pyautogui.write(pressed_key.lower())
                          self.typed_text += pressed_key
                                    
                                    # Highlight pressed key
                        for key, x, y, w_key, h_key in button_list:
                          if key == pressed_key:
                            cv2.rectangle(img, (x, y), (x + w_key, y + h_key), 
                                                        (0, 255, 0), -1)
                                    
                          self.last_click_time = current_time
            
        cv2.imshow("Virtual Keyboard & Mouse", img)
            
        key = cv2.waitKey(1)
        if key == ord('q'):
          break
        elif key == ord('k'):
          self.mode = "KEYBOARD"
          self.typed_text = ""
        elif key == ord('m'):
          self.mode = "MOUSE"
        
      cap.release()
      cv2.destroyAllWindows()  


if __name__ == "__main__":
    app = VirtualKeyboardMouse()
    app.run()