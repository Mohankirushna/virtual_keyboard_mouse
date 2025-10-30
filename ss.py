import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from PIL import ImageGrab

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
        self.last_rock_gesture_time = 0  # Separate cooldown for rock gesture
        
        # Drawing variables
        self.canvas = None
        self.draw_color = (255, 0, 0)  # Blue color for drawing
        self.draw_thickness = 5
        self.prev_draw_point = None
        self.color_options = [
            ((255, 0, 0), "Blue"),
            ((0, 255, 0), "Green"),
            ((0, 0, 255), "Red"),
            ((0, 255, 255), "Yellow"),
            ((255, 0, 255), "Magenta"),
            ((0, 0, 0), "Black"),
            ((255, 255, 255), "White")
        ]
        self.selected_color_idx = 0
        
        # Keyboard layout
        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.'],
            ['SPACE', 'BACKSPACE', 'ENTER']
        ]
        
        self.mode = "MOUSE"  # MOUSE, KEYBOARD, or DRAW
        self.typed_text = ""
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0
        
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
    
    def draw_color_palette(self, img):
        """Draw color palette for drawing mode"""
        h, w = img.shape[:2]
        palette_y = h - 100
        palette_x = 50
        box_size = 50
        margin = 10
        
        overlay = img.copy()
        alpha = 0.7
        
        for i, (color, name) in enumerate(self.color_options):
            x = palette_x + i * (box_size + margin)
            cv2.rectangle(overlay, (x, palette_y), (x + box_size, palette_y + box_size), color, -1)
            
            # Highlight selected color
            if i == self.selected_color_idx:
                cv2.rectangle(overlay, (x - 3, palette_y - 3), (x + box_size + 3, palette_y + box_size + 3), (0, 255, 0), 3)
        
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw toolbar
        cv2.rectangle(img, (palette_x, palette_y - 40), (palette_x + 200, palette_y - 10), (40, 40, 40), -1)
        cv2.putText(img, "CLEAR", (palette_x + 10, palette_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "SAVE", (palette_x + 110, palette_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return [(palette_x, palette_y - 40, 90, 30), (palette_x + 100, palette_y - 40, 90, 30)]
    
    def check_key_press(self, finger_tip, button_list):
        """Check if finger is pressing a key"""
        for key, x, y, w, h in button_list:
            if x < finger_tip[0] < x + w and y < finger_tip[1] < y + h:
                return key
        return None
    
    def check_toolbar_press(self, finger_tip, toolbar_list):
        """Check if finger is pressing a toolbar button"""
        for i, (x, y, w, h) in enumerate(toolbar_list):
            if x < finger_tip[0] < x + w and y < finger_tip[1] < y + h:
                return i  # 0 for CLEAR, 1 for SAVE
        return None

    def is_fist(self, lm_list):
        """Returns True if all fingers are folded (fist gesture)"""
        if len(lm_list) < 21:
            return False
        folded = 0
        for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            if lm_list[tip][1] > lm_list[pip][1]:  # y increases downward
                folded += 1
        # Thumb: check if tip is close to palm
        thumb_folded = abs(lm_list[4][0] - lm_list[2][0]) < 40
        return folded == 4 and thumb_folded
    
    def is_peace_sign(self, lm_list):
        """Returns True if index and middle fingers are up (peace/victory sign)"""
        if len(lm_list) < 21:
            return False
        
        # Index and middle up
        index_up = lm_list[8][1] < lm_list[6][1]
        middle_up = lm_list[12][1] < lm_list[10][1]
        
        # Ring and pinky down
        ring_down = lm_list[16][1] > lm_list[14][1]
        pinky_down = lm_list[20][1] > lm_list[18][1]
        
        return index_up and middle_up and ring_down and pinky_down
    
    def is_thumbs_up(self, lm_list):
        """Returns True if thumb is up and other fingers are folded"""
        if len(lm_list) < 21:
            return False
        
        # Thumb up (y-coordinate of tip is less than base)
        thumb_up = lm_list[4][1] < lm_list[2][1]
        
        # Other fingers folded
        fingers_folded = 0
        for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            if lm_list[tip][1] > lm_list[pip][1]:
                fingers_folded += 1
        
        return thumb_up and fingers_folded >= 3
    
    def is_thumbs_down(self, lm_list):
        """Returns True if thumb is down and other fingers are folded"""
        if len(lm_list) < 21:
            return False
        
        # Thumb down (y-coordinate of tip is greater than base)
        thumb_down = lm_list[4][1] > lm_list[2][1]
        
        # Other fingers folded
        fingers_folded = 0
        for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            if lm_list[tip][1] > lm_list[pip][1]:
                fingers_folded += 1
        
        return thumb_down and fingers_folded >= 3
    
    def is_pointing(self, lm_list):
        """Returns True if only index finger is up"""
        if len(lm_list) < 21:
            return False
        
        # Index up
        index_up = lm_list[8][1] < lm_list[6][1]
        
        # Other fingers down
        middle_down = lm_list[12][1] > lm_list[10][1]
        ring_down = lm_list[16][1] > lm_list[14][1]
        pinky_down = lm_list[20][1] > lm_list[18][1]
        
        return index_up and middle_down and ring_down and pinky_down
    
    def is_two_fingers_pinch(self, lm_list):
        """Returns True if index and middle fingers are pinched together (for double click)"""
        if len(lm_list) < 21:
            return False
        
        # Distance between index and middle fingertips
        index_tip = lm_list[8]
        middle_tip = lm_list[12]
        dist = self.distance(index_tip, middle_tip)
        
        # Both fingers should be extended
        index_up = lm_list[8][1] < lm_list[6][1]
        middle_up = lm_list[12][1] < lm_list[10][1]
        
        # Ring and pinky should be down
        ring_down = lm_list[16][1] > lm_list[14][1]
        pinky_down = lm_list[20][1] > lm_list[18][1]
        
        return dist < 50 and index_up and middle_up and ring_down and pinky_down
    
    def is_three_fingers_up(self, lm_list):
        """Returns True if index, middle, and ring fingers are up (alternative double click)"""
        if len(lm_list) < 21:
            return False
        
        # Index, middle, and ring up
        index_up = lm_list[8][1] < lm_list[6][1]
        middle_up = lm_list[12][1] < lm_list[10][1]
        ring_up = lm_list[16][1] < lm_list[14][1]
        
        # Pinky down
        pinky_down = lm_list[20][1] > lm_list[18][1]
        
        return index_up and middle_up and ring_up and pinky_down
    
    def is_rock_gesture(self, lm_list):
        """Returns True if index and pinky are up, middle and ring are down (rock on 🤘)"""
        if len(lm_list) < 21:
            return False
        
        # Index and pinky up (y-coordinate comparison - lower y means higher up)
        index_up = lm_list[8][1] < lm_list[6][1] - 20  # More strict
        pinky_up = lm_list[20][1] < lm_list[18][1] - 20  # More strict
        
        # Middle and ring down (clearly folded)
        middle_down = lm_list[12][1] > lm_list[10][1] + 10
        ring_down = lm_list[16][1] > lm_list[14][1] + 10
        
        is_rock = index_up and pinky_up and middle_down and ring_down
        
        # Debug: print when rock gesture is detected
        if is_rock:
            print("🤘 ROCK GESTURE DETECTED!")
        
        return is_rock

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
        print("- Press 'D' to switch to Draw mode")
        print("- Press 'Q' to quit")
        print("\nMouse Mode:")
        print("- Move index finger to control cursor")
        print("- Pinch index and thumb to single click")
        print("- Rock gesture 🤘 (index+pinky up) for double click - OPENS FOLDERS!")
        print("\nKeyboard Mode:")
        print("- Point at keys to type")
        print("- Pinch index and thumb to press key")
        print("\nDraw Mode:")
        print("- Point with index finger to draw")
        print("- Make a fist to stop drawing")
        print("- Pinch to select colors or use toolbar")
        print("\nGestures (work in all modes):")
        print("- Fist: Toggle between modes")
        print("- Peace sign ✌: Take screenshot")
        print("- Rock gesture 🤘: Double-click (opens folders)")
        print("- Thumbs up 👍: Scroll up")
        print("- Thumbs down 👎: Scroll down")
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            # Initialize canvas if in draw mode
            if self.mode == "DRAW" and self.canvas is None:
                self.canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_img)
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
                frame_count = 0
            
            # Draw mode indicator
            mode_colors = {"MOUSE": (0, 255, 0), "KEYBOARD": (0, 165, 255), "DRAW": (255, 0, 255)}
            mode_color = mode_colors.get(self.mode, (255, 255, 255))
            cv2.putText(img, f"Mode: {self.mode}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
            cv2.putText(img, f"FPS: {int(fps)}", (w - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Always define button_list and toolbar_list
            button_list = None
            toolbar_list = None
            if self.mode == "KEYBOARD":
                button_list = self.draw_keyboard(img)
            elif self.mode == "DRAW":
                toolbar_list = self.draw_color_palette(img)
                # Overlay canvas on image
                mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)
                mask_inv = cv2.bitwise_not(mask)
                img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
                canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
                img = cv2.add(img_bg, canvas_fg)
            
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_draw.draw_landmarks(img, hand_landmarks, 
                                                self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get landmarks
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        lm_list.append([int(lm.x * w), int(lm.y * h)])
                    
                    # Get handedness
                    handedness = None
                    if hasattr(results, 'multi_handedness') and results.multi_handedness:
                        handedness = results.multi_handedness[hand_idx].classification[0].label
                        is_left = handedness == "Left"
                    
                    current_time = time.time()
                    
                    # Check for rock gesture FIRST (highest priority for double-click)
                    rock_detected = self.is_rock_gesture(lm_list)
                    
                    # Gesture detection for mode switching and actions
                    if current_time - self.last_gesture_time > self.gesture_cooldown:
                        # Fist to toggle modes
                        if is_left and self.is_fist(lm_list):
                            modes = ["MOUSE", "KEYBOARD", "DRAW"]
                            current_idx = modes.index(self.mode)
                            self.mode = modes[(current_idx + 1) % len(modes)]
                            self.last_gesture_time = current_time
                            if self.mode == "DRAW" and self.canvas is None:
                                self.canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
                        
                        # Peace sign for screenshot
                        elif self.is_peace_sign(lm_list):
                            screenshot = ImageGrab.grab()
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            filename = f"screenshot_{timestamp}.png"
                            screenshot.save(filename)
                            cv2.putText(img, f"Screenshot saved: {filename}", (w//2 - 300, h//2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            self.last_gesture_time = current_time
                            print(f"Screenshot saved: {filename}")
                        
                        # Thumbs up for scroll up
                        elif self.is_thumbs_up(lm_list):
                            pyautogui.scroll(300)
                            cv2.putText(img, "Scrolling Up", (w//2 - 150, h//2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            self.last_gesture_time = current_time
                        
                        # Thumbs down for scroll down
                        elif self.is_thumbs_down(lm_list):
                            pyautogui.scroll(-300)
                            cv2.putText(img, "Scrolling Down", (w//2 - 150, h//2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            self.last_gesture_time = current_time
                    
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
                            screen_x = np.interp(index_tip[0], [100, w-100], [0, self.screen_w])
                            screen_y = np.interp(index_tip[1], [100, h-100], [0, self.screen_h])
                            
                            # Smooth cursor movement
                            curr_x = self.prev_x + (screen_x - self.prev_x) / self.smoothening
                            curr_y = self.prev_y + (screen_y - self.prev_y) / self.smoothening
                            
                            pyautogui.moveTo(curr_x, curr_y)
                            self.prev_x, self.prev_y = curr_x, curr_y
                            
                            # PRIORITY: Rock gesture for instant double-click
                            if rock_detected and (current_time - self.last_rock_gesture_time > 0.5):
                                cv2.circle(img, tuple(index_tip), 20, (255, 0, 255), -1)
                                cv2.putText(img, "DOUBLE CLICK! 🤘", (index_tip[0] - 120, index_tip[1] - 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
                                pyautogui.doubleClick(interval=0.1)
                                self.last_rock_gesture_time = current_time
                                print("🤘 DOUBLE CLICK performed (rock gesture)")
                            
                            # Single click (thumb + index pinch)
                            elif dist < self.click_threshold:
                                cv2.circle(img, tuple(index_tip), 15, (0, 255, 0), -1)
                                if current_time - self.last_click_time > self.click_cooldown:
                                    pyautogui.click()
                                    self.last_click_time = current_time
                        
                        elif self.mode == "KEYBOARD":
                            if button_list is not None:
                                pressed_key = self.check_key_press(index_tip, button_list)
                                
                                if pressed_key and dist < self.click_threshold:
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
                        
                        elif self.mode == "DRAW":
                            # Check if pointing (only index finger up)
                            if self.is_pointing(lm_list):
                                # Draw on canvas
                                if self.prev_draw_point is not None:
                                    cv2.line(self.canvas, self.prev_draw_point, tuple(index_tip), 
                                            self.draw_color, self.draw_thickness)
                                self.prev_draw_point = tuple(index_tip)
                            else:
                                self.prev_draw_point = None
                            
                            # Check for toolbar interaction with pinch
                            if toolbar_list is not None and dist < self.click_threshold:
                                if current_time - self.last_click_time > self.click_cooldown:
                                    toolbar_action = self.check_toolbar_press(index_tip, toolbar_list)
                                    if toolbar_action == 0:  # CLEAR
                                        self.canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
                                        self.last_click_time = current_time
                                    elif toolbar_action == 1:  # SAVE
                                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                                        filename = f"drawing_{timestamp}.png"
                                        cv2.imwrite(filename, self.canvas)
                                        cv2.putText(img, f"Drawing saved: {filename}", (w//2 - 300, 100), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                        self.last_click_time = current_time
                                        print(f"Drawing saved: {filename}")
                                    
                                    # Check color selection
                                    palette_y = h - 100
                                    palette_x = 50
                                    box_size = 50
                                    margin = 10
                                    for i in range(len(self.color_options)):
                                        x = palette_x + i * (box_size + margin)
                                        if x < index_tip[0] < x + box_size and palette_y < index_tip[1] < palette_y + box_size:
                                            self.selected_color_idx = i
                                            self.draw_color = self.color_options[i][0]
                                            self.last_click_time = current_time
                                            break
            
            cv2.imshow("Virtual Keyboard, Mouse & Drawing", img)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('k'):
                self.mode = "KEYBOARD"
                self.typed_text = ""
            elif key == ord('m'):
                self.mode = "MOUSE"
            elif key == ord('d'):
                self.mode = "DRAW"
                if self.canvas is None:
                    _, img_temp = cap.read()
                    h, w = img_temp.shape[:2]
                    self.canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = VirtualKeyboardMouse()
    app.run()