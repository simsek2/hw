import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Run webcam capture
cap = cv2.VideoCapture(0)

# Button with 3 states, default, hover, and click
button1 = {'pos': [100, 200], 'size': (150, 150), 'default_color': (200, 0, 0), 'hover_color': (0, 255, 0),
           'click_color': (0, 0, 255), 'current_color': (200, 0, 0), 'label': 'Left Button', 'hovered': False,
           'name': 'Button'}

# Scrollbars with an additional moving color state
v_scrollbar = {'x': 600, 'y': 200, 'width': 50, 'height': 150, 'default_color': (150, 150, 150),
               'hover_color': (100, 100, 100), 'moving_color': (80, 80, 80), 'current_color': (150, 150, 150)}

v_scrollbar2 = {'x': 0, 'y': 200, 'width': 50, 'height': 150, 'default_color': (150, 150, 150),
               'hover_color': (100, 100, 100), 'moving_color': (80, 80, 80), 'current_color': (150, 150, 150)}

h_scrollbar = {'x': 100, 'y': 0, 'width': 150, 'height': 50, 'default_color': (150, 150, 150),
               'hover_color': (100, 100, 100), 'moving_color': (80, 80, 80), 'current_color': (150, 150, 150)}

h_scrollbar2 = {'x': 100, 'y': 430, 'width': 150, 'height': 50, 'default_color': (150, 150, 150),
               'hover_color': (100, 100, 100), 'moving_color': (80, 80, 80), 'current_color': (150, 150, 150)}

# Screen dimensions
screen_width, screen_height = 640, 480

clicked_button = ""

# For tracking initial positions for scroll interaction
initial_finger_positions = {'h_scrollbar': None, 'v_scrollbar': None, 'h_scrollbar2': None, 'v_scrollbar2': None}


# Load the image
image = cv2.imread("monitor.png")
small_image_size = (300, 200)
small_image = cv2.resize(image, small_image_size)

# Position for the small image
small_image_pos = (screen_width - small_image_size[0] - 5, 5)

# Full-screen size
full_screen_image_size = (screen_width, screen_height)
full_screen_image = cv2.resize(image, full_screen_image_size)

is_over_image = False


# Function to draw an arrow at the end of a scrollbar
def draw_arrow(frame, start_point, direction, color=(255, 255, 255), thickness=2):
    """Draws an arrow at the end of a scrollbar."""
    length = 20  # Length of the arrow
    if direction == 'up':
        end_point = (start_point[0], start_point[1] - length)
    elif direction == 'down':
        end_point = (start_point[0], start_point[1] + length)
    elif direction == 'left':
        end_point = (start_point[0] - length, start_point[1])
    elif direction == 'right':
        end_point = (start_point[0] + length, start_point[1])
    else:
        return
    cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=0.5)


with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        x, y = small_image_pos
        frame[y:y + small_image_size[1], x:x + small_image_size[0]] = small_image

        # Draw Button and Label
        for button in [button1]:
            # Draw button rectangle
            cv2.rectangle(frame, tuple(button['pos']),
                          (button['pos'][0] + button['size'][0], button['pos'][1] + button['size'][1]),
                          button['current_color'], -1)

            # Draw button label
            text_x = button['pos'][0] + 40
            text_y = button['pos'][1] + button['size'][1] // 2 + 10
            cv2.putText(frame, button['name'], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw scrollbars with their current colors
        cv2.rectangle(frame, (v_scrollbar['x'], v_scrollbar['y']),
                      (v_scrollbar['x'] + v_scrollbar['width'], v_scrollbar['y'] + v_scrollbar['height']),
                      v_scrollbar['current_color'], -1)
        cv2.rectangle(frame, (h_scrollbar['x'], h_scrollbar['y']),
                      (h_scrollbar['x'] + h_scrollbar['width'], h_scrollbar['y'] + h_scrollbar['height']),
                      h_scrollbar['current_color'], -1)

        cv2.rectangle(frame, (v_scrollbar2['x'], v_scrollbar2['y']),
                      (v_scrollbar2['x'] + v_scrollbar2['width'], v_scrollbar2['y'] + v_scrollbar2['height']),
                      v_scrollbar2['current_color'], -1)
        cv2.rectangle(frame, (h_scrollbar2['x'], h_scrollbar2['y']),
                      (h_scrollbar2['x'] + h_scrollbar2['width'], h_scrollbar2['y'] + h_scrollbar2['height']),
                      h_scrollbar2['current_color'], -1)

        is_h_scrollbar_drawn = False
        is_h_scrollbar2_drawn = False
        is_v_scrollbar_drawn = False
        is_v_scrollbar2_drawn = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get fingertip positions
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Cursor positions are based on index finger tip
                cursor_x = int(index_tip.x * screen_width)
                cursor_y = int(index_tip.y * screen_height)

                is_over_image = (x <= cursor_x <= x + small_image_size[0] and
                                 y <= cursor_y <= y + small_image_size[1])

                if is_over_image:
                    frame = full_screen_image.copy()
                    break

                # Draw virtual cursor
                cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 0, 0), -1)

                # Check if cursor (index tip) is over any button
                is_over_button1 = button1['pos'][0] <= cursor_x <= button1['pos'][0] + button1['size'][0] and button1['pos'][1] <= cursor_y <= button1['pos'][1] + button1['size'][1]
                
                # Detect "pinch" gesture for clicking
                distance_thumb_index = np.linalg.norm(
                    [thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]
                )

                pinch_threshold = 0.05

                # Button hover and click states for button1
                if is_over_button1:
                    if distance_thumb_index < pinch_threshold:
                        # Red when clicked
                        button1['current_color'] = button1['click_color']
                        clicked_button = button1['label']
                        # cv2.putText(frame, f"{clicked_button} Clicked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
                    else:
                        # Green when hovered
                        button1['current_color'] = button1['hover_color']
                else:
                    # Reset to default color
                    button1['current_color'] = button1['default_color']

                # Check if index tip is over any scrollbar
                is_over_v_scrollbar = v_scrollbar['x'] <= cursor_x <= v_scrollbar['x'] + v_scrollbar['width'] and v_scrollbar['y'] <= cursor_y <= v_scrollbar['y'] + v_scrollbar['height']
                is_over_h_scrollbar = h_scrollbar['x'] <= cursor_x <= h_scrollbar['x'] + h_scrollbar['width'] and h_scrollbar['y'] <= cursor_y <= h_scrollbar['y'] + h_scrollbar['height']

                # Check if index tip is over any scrollbar
                is_over_v_scrollbar2 = v_scrollbar2['x'] <= cursor_x <= v_scrollbar2['x'] + v_scrollbar2['width'] and v_scrollbar2['y'] <= cursor_y <= v_scrollbar2['y'] + v_scrollbar2['height']
                is_over_h_scrollbar2 = h_scrollbar2['x'] <= cursor_x <= h_scrollbar2['x'] + h_scrollbar2['width'] and h_scrollbar2['y'] <= cursor_y <= h_scrollbar2['y'] + h_scrollbar2['height']

                middle_x = int(middle_tip.x * screen_width)
                middle_y = int(middle_tip.y * screen_height)

                # Check if middle tip is over any scrollbar
                middle_over_v_scrollbar = v_scrollbar['x'] <= middle_x <= v_scrollbar['x'] + v_scrollbar['width'] and v_scrollbar['y'] <= middle_y <= v_scrollbar['y'] + v_scrollbar['height']
                middle_over_h_scrollbar = h_scrollbar['x'] <= middle_x <= h_scrollbar['x'] + h_scrollbar['width'] and h_scrollbar['y'] <= middle_y <= h_scrollbar['y'] + h_scrollbar['height']

                middle_over_v_scrollbar2 = v_scrollbar2['x'] <= middle_x <= v_scrollbar2['x'] + v_scrollbar2['width'] and v_scrollbar2['y'] <= middle_y <= v_scrollbar2['y'] + v_scrollbar2['height']
                middle_over_h_scrollbar2 = h_scrollbar2['x'] <= middle_x <= h_scrollbar2['x'] + h_scrollbar2['width'] and h_scrollbar2['y'] <= middle_y <= h_scrollbar2['y'] + h_scrollbar2['height']

                # Scrollbars move based on finger movement
                if is_over_h_scrollbar and middle_over_h_scrollbar:
                    if initial_finger_positions['h_scrollbar'] is None:
                        initial_finger_positions['h_scrollbar'] = (cursor_x, middle_x)

                    delta_x = cursor_x - initial_finger_positions['h_scrollbar'][0]
                    h_scrollbar['x'] += delta_x
                    button1['pos'][0] += delta_x

                    # Set scrollbar color to moving color when scrolling
                    h_scrollbar['current_color'] = h_scrollbar['moving_color']

                    if delta_x >= 2:
                        # Moving to the right
                        draw_arrow(frame, (
                        h_scrollbar['x'] + h_scrollbar['width'], h_scrollbar['y'] + h_scrollbar['height'] // 2),
                                   'right', color=(0, 0, 0))
                        draw_arrow(frame, (h_scrollbar['x'], h_scrollbar['y'] + h_scrollbar['height'] // 2), 'left',
                                   color=(255, 255, 255))
                    elif delta_x <= -2:
                        # Moving to the left
                        draw_arrow(frame, (
                        h_scrollbar['x'] + h_scrollbar['width'], h_scrollbar['y'] + h_scrollbar['height'] // 2),
                                   'right', color=(255, 255, 255))
                        draw_arrow(frame, (h_scrollbar['x'], h_scrollbar['y'] + h_scrollbar['height'] // 2), 'left',
                                   color=(0, 0, 0))
                    else:
                        draw_arrow(frame, (
                        h_scrollbar['x'] + h_scrollbar['width'], h_scrollbar['y'] + h_scrollbar['height'] // 2),
                                   'right', color=(255, 255, 255))
                        draw_arrow(frame, (h_scrollbar['x'], h_scrollbar['y'] + h_scrollbar['height'] // 2), 'left',
                                   color=(255, 255, 255))

                    initial_finger_positions['h_scrollbar'] = (cursor_x, middle_x)
                    is_h_scrollbar_drawn = True
                elif is_over_h_scrollbar:
                    # Set to hover color when hovered
                    h_scrollbar['current_color'] = h_scrollbar['hover_color']
                    initial_finger_positions['h_scrollbar'] = None
                else:
                    # Reset color when not hovered or moving
                    h_scrollbar['current_color'] = h_scrollbar['default_color']
                    initial_finger_positions['h_scrollbar'] = None


                if is_over_h_scrollbar2 and middle_over_h_scrollbar2:
                    if initial_finger_positions['h_scrollbar2'] is None:
                        initial_finger_positions['h_scrollbar2'] = (cursor_x, middle_x)

                    delta_x = cursor_x - initial_finger_positions['h_scrollbar2'][0]
                    h_scrollbar2['x'] += delta_x
                    button1['pos'][0] += delta_x

                    # Set scrollbar color to moving color when scrolling
                    h_scrollbar2['current_color'] = h_scrollbar2['moving_color']

                    if delta_x >= 2:
                        # Moving to the right
                        draw_arrow(frame, (
                        h_scrollbar2['x'] + h_scrollbar2['width'], h_scrollbar2['y'] + h_scrollbar2['height'] // 2),
                                   'right', color=(0, 0, 0))
                        draw_arrow(frame, (h_scrollbar2['x'], h_scrollbar2['y'] + h_scrollbar2['height'] // 2), 'left',
                                   color=(255, 255, 255))
                    elif delta_x <= -2:
                        # Moving to the left
                        draw_arrow(frame, (
                        h_scrollbar2['x'] + h_scrollbar2['width'], h_scrollbar2['y'] + h_scrollbar2['height'] // 2),
                                   'right', color=(255, 255, 255))
                        draw_arrow(frame, (h_scrollbar2['x'], h_scrollbar2['y'] + h_scrollbar2['height'] // 2), 'left',
                                   color=(0, 0, 0))
                    else:
                        draw_arrow(frame, (
                        h_scrollbar2['x'] + h_scrollbar2['width'], h_scrollbar2['y'] + h_scrollbar2['height'] // 2),
                                   'right', color=(255, 255, 255))
                        draw_arrow(frame, (h_scrollbar2['x'], h_scrollbar2['y'] + h_scrollbar2['height'] // 2), 'left',
                                   color=(255, 255, 255))

                    initial_finger_positions['h_scrollbar2'] = (cursor_x, middle_x)
                    is_h_scrollbar2_drawn = True
                elif is_over_h_scrollbar2:
                    # Set to hover color when hovered
                    h_scrollbar2['current_color'] = h_scrollbar2['hover_color']
                    initial_finger_positions['h_scrollbar2'] = None
                else:
                    # Reset color when not hovered or moving
                    h_scrollbar2['current_color'] = h_scrollbar2['default_color']
                    initial_finger_positions['h_scrollbar2'] = None

                if is_over_v_scrollbar and middle_over_v_scrollbar:
                    if initial_finger_positions['v_scrollbar'] is None:
                        initial_finger_positions['v_scrollbar'] = (cursor_y, middle_y)

                    delta_y = cursor_y - initial_finger_positions['v_scrollbar'][0]
                    v_scrollbar['y'] += delta_y
                    button1['pos'][1] += delta_y

                    # Set scrollbar color to moving color when scrolling
                    v_scrollbar['current_color'] = v_scrollbar['moving_color']

                    if delta_y >= 2:
                        # Moving down
                        draw_arrow(frame, (v_scrollbar['x'] + v_scrollbar['width'] // 2, v_scrollbar['y']), 'up',
                                   color=(255, 255, 255))
                        draw_arrow(frame, (
                        v_scrollbar['x'] + v_scrollbar['width'] // 2, v_scrollbar['y'] + v_scrollbar['height']), 'down',
                                   color=(0, 0, 0))
                    elif delta_y <= -2:
                        # Moving up
                        draw_arrow(frame, (v_scrollbar['x'] + v_scrollbar['width'] // 2, v_scrollbar['y']), 'up',
                                   color=(0, 0, 0))
                        draw_arrow(frame, (
                        v_scrollbar['x'] + v_scrollbar['width'] // 2, v_scrollbar['y'] + v_scrollbar['height']), 'down',
                                   color=(255, 255, 255))
                    else:
                        draw_arrow(frame, (v_scrollbar['x'] + v_scrollbar['width'] // 2, v_scrollbar['y']), 'up',
                                   color=(255, 255, 255))
                        draw_arrow(frame, (
                            v_scrollbar['x'] + v_scrollbar['width'] // 2, v_scrollbar['y'] + v_scrollbar['height']),
                                   'down',
                                   color=(255, 255, 255))

                    initial_finger_positions['v_scrollbar'] = (cursor_y, middle_y)
                    is_v_scrollbar_drawn = True
                elif is_over_v_scrollbar:
                    # Set to hover color when hovered
                    v_scrollbar['current_color'] = v_scrollbar['hover_color']
                    initial_finger_positions['v_scrollbar'] = None
                else:
                    # Reset color when not hovered or moving
                    v_scrollbar['current_color'] = v_scrollbar['default_color']
                    initial_finger_positions['v_scrollbar'] = None

            if is_over_v_scrollbar2 and middle_over_v_scrollbar2:
                if initial_finger_positions['v_scrollbar2'] is None:
                    initial_finger_positions['v_scrollbar2'] = (cursor_y, middle_y)

                delta_y = cursor_y - initial_finger_positions['v_scrollbar2'][0]
                v_scrollbar2['y'] += delta_y
                button1['pos'][1] += delta_y

                # Set scrollbar color to moving color when scrolling
                v_scrollbar2['current_color'] = v_scrollbar2['moving_color']

                if delta_y >= 2:
                    # Moving down
                    draw_arrow(frame, (v_scrollbar2['x'] + v_scrollbar2['width'] // 2, v_scrollbar2['y']), 'up',
                               color=(255, 255, 255))
                    draw_arrow(frame, (
                        v_scrollbar2['x'] + v_scrollbar2['width'] // 2, v_scrollbar2['y'] + v_scrollbar2['height']), 'down',
                               color=(0, 0, 0))
                elif delta_y <= -2:
                    # Moving up
                    draw_arrow(frame, (v_scrollbar2['x'] + v_scrollbar2['width'] // 2, v_scrollbar2['y']), 'up',
                               color=(0, 0, 0))
                    draw_arrow(frame, (
                        v_scrollbar2['x'] + v_scrollbar2['width'] // 2, v_scrollbar2['y'] + v_scrollbar2['height']), 'down',
                               color=(255, 255, 255))
                else:
                    draw_arrow(frame, (v_scrollbar2['x'] + v_scrollbar2['width'] // 2, v_scrollbar2['y']), 'up',
                               color=(255, 255, 255))
                    draw_arrow(frame, (
                        v_scrollbar2['x'] + v_scrollbar2['width'] // 2, v_scrollbar2['y'] + v_scrollbar2['height']),
                               'down',
                               color=(255, 255, 255))

                initial_finger_positions['v_scrollbar2'] = (cursor_y, middle_y)
                is_v_scrollbar2_drawn = True
            elif is_over_v_scrollbar2:
                # Set to hover color when hovered
                v_scrollbar2['current_color'] = v_scrollbar2['hover_color']
                initial_finger_positions['v_scrollbar2'] = None
            else:
                # Reset color when not hovered or moving
                v_scrollbar2['current_color'] = v_scrollbar2['default_color']
                initial_finger_positions['v_scrollbar2'] = None

        if not is_h_scrollbar_drawn:
            draw_arrow(frame, (h_scrollbar['x'] + h_scrollbar['width'], h_scrollbar['y'] + h_scrollbar['height'] // 2),
                       'right', color=(255, 255, 255))
            draw_arrow(frame, (h_scrollbar['x'], h_scrollbar['y'] + h_scrollbar['height'] // 2), 'left',
                   color=(255, 255, 255))

        if not is_h_scrollbar2_drawn:
            draw_arrow(frame, (h_scrollbar2['x'] + h_scrollbar2['width'], h_scrollbar2['y'] + h_scrollbar2['height'] // 2),
                       'right', color=(255, 255, 255))
            draw_arrow(frame, (h_scrollbar2['x'], h_scrollbar2['y'] + h_scrollbar2['height'] // 2), 'left',
                       color=(255, 255, 255))

        if not is_v_scrollbar_drawn:
            draw_arrow(frame, (v_scrollbar['x'] + v_scrollbar['width'] // 2, v_scrollbar['y']), 'up',
                       color=(255, 255, 255))
            draw_arrow(frame, (
                v_scrollbar['x'] + v_scrollbar['width'] // 2, v_scrollbar['y'] + v_scrollbar['height']), 'down',
                       color=(255, 255, 255))

        if not is_v_scrollbar2_drawn:
            draw_arrow(frame, (v_scrollbar2['x'] + v_scrollbar2['width'] // 2, v_scrollbar2['y']), 'up',
                       color=(255, 255, 255))
            draw_arrow(frame, (
                v_scrollbar2['x'] + v_scrollbar2['width'] // 2, v_scrollbar2['y'] + v_scrollbar2['height']), 'down',
                       color=(255, 255, 255))

        cv2.imshow('Gesture-Based Interface', frame)

        # Press 'q' to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
