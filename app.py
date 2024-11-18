import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
from flask import Flask, render_template, Response

app = Flask(__name__)
kyc_started = False #for the state of the button


BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20


user_response = None
button_pressed = False
waiting_for_response = True
action_sequence = []
liveness_confirmed = False


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)


MOUTH = [13, 14, 15, 17, 18, 19, 20, 22]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
def detect_head_movement(landmarks, initial_nose_point):#head movement using nose landmarks 
    current_nose_point = landmarks[1] 
    dx = current_nose_point[0] - initial_nose_point[0] #horizontal displacement
    dy = current_nose_point[1] - initial_nose_point[1] #vertical displacement
    return dx, dy

def get_eye_center(landmarks, eye_indices):#for follow dot test ,the center eye landmark is calc
    x = [landmarks[i][0] for i in eye_indices]
    y = [landmarks[i][1] for i in eye_indices]
    center = (int(sum(x) / len(x)), int(sum(y) / len(y)))
    return center

def calculate_ear(landmarks, eye_indices):
    A = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
    return (A + B) / (2.0 * C)


def calculate_mar(landmarks, mouth_indices):
    A = np.linalg.norm(np.array(landmarks[mouth_indices[0]]) - np.array(landmarks[mouth_indices[4]]))
    B = np.linalg.norm(np.array(landmarks[mouth_indices[1]]) - np.array(landmarks[mouth_indices[5]]))
    C = np.linalg.norm(np.array(landmarks[mouth_indices[2]]) - np.array(landmarks[mouth_indices[6]]))
    D = np.linalg.norm(np.array(landmarks[mouth_indices[3]]) - np.array(landmarks[mouth_indices[7]]))
    return (A + B + C) / (2.0 * D)


def get_button_positions(frame_width, frame_height):
    yes_button_top_left = (BUTTON_MARGIN, frame_height - BUTTON_HEIGHT - BUTTON_MARGIN)
    yes_button_bottom_right = (BUTTON_MARGIN + BUTTON_WIDTH, frame_height - BUTTON_MARGIN)
    no_button_top_left = (BUTTON_MARGIN * 2 + BUTTON_WIDTH, frame_height - BUTTON_HEIGHT - BUTTON_MARGIN)
    no_button_bottom_right = (BUTTON_MARGIN * 2 + BUTTON_WIDTH * 2, frame_height - BUTTON_MARGIN)
    return (yes_button_top_left, yes_button_bottom_right, no_button_top_left, no_button_bottom_right)


def mouse_callback(event, x, y, flags, param):
    global user_response, button_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        (yes_tl, yes_br, no_tl, no_br) = param
        if yes_tl[0] <= x <= yes_br[0] and yes_tl[1] <= y <= yes_br[1]:
            user_response = 'yes'
            button_pressed = True
        elif no_tl[0] <= x <= no_br[0] and no_tl[1] <= y <= no_br[1]:
            user_response = 'no'
            button_pressed = True

# Video capture and streaming
def generate():
    global kyc_started
    if not kyc_started:
        return
    global user_response, button_pressed, waiting_for_response, action_sequence
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    cv2.namedWindow('KYC Prompt')  # Creating the window here
    frame_height, frame_width = cap.read()[1].shape[:2]  # Get the video frame size
    yes_tl, yes_br, no_tl, no_br = get_button_positions(frame_width, frame_height)
    cv2.setMouseCallback('KYC Prompt', mouse_callback, param=(yes_tl, yes_br, no_tl, no_br))
    liveness_confirmed = False #overall flag
    action_failed= False
    mp_face_mesh = mp.solutions.face_mesh #landmark analysis
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    #pre-defined facemarks
    MOUTH = [13, 14, 15, 17, 18, 19, 20, 22]
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    NOSE_TIP = 1

    actions = ['smile', 'turn left', 'turn right', 'look up', 'look down']#'blink'
    action_detected = False
    action_sequence = []

    mar_threshold = 0.6#needs to achieve >60%  
    mar_smoothing = 5# prevents rapid changes in MAR from trigerring action detected and 5 is the number of MAR values to average
    mar_history = []

    waiting_for_response = True

    gaze_points = []
    gaze_index = 0
    gaze_point = None
    gaze_start_time = None
    gaze_duration = 3  # in seconds
    gaze_movement_time = 1.5  # in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        yes_tl, yes_br, no_tl, no_br = get_button_positions(frame_width, frame_height)

        if waiting_for_response:
            cv2.putText(frame, "Are you ready for KYC?", (BUTTON_MARGIN, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, yes_tl, yes_br, (0, 255, 0), -1)
            cv2.putText(frame, "YES", (yes_tl[0] + 20, yes_tl[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(frame, no_tl, no_br, (0, 0, 255), -1)
            cv2.putText(frame, "NO", (no_tl[0] + 25, no_tl[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('KYC Prompt', frame)

          
            if button_pressed:
                if user_response == 'yes':
                    waiting_for_response = False
                    selected_actions = random.sample(actions, 2) #any random actions 
                    action_sequence = selected_actions + ['follow dot'] #making follow dot a necessary one
                    print(f"Please perform the following actions in order:")
                    for idx, action in enumerate(action_sequence):
                        print(f"{idx + 1}. {action.upper()}")#action sequence iteration
                elif user_response == 'no':
                    print("User chose NO. Exiting KYC process.")
                    break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

        else:
            # Process the frame with MediaPipe for face landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [(int(point.x * frame_width), int(point.y * frame_height)) for point in face_landmarks.landmark]
                if 'initial_nose_point' not in locals():#already defined 
                    initial_nose_point = landmarks[NOSE_TIP]

                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0 

                mar = calculate_mar(landmarks, MOUTH)
                mar_history.append(mar) # takes 5 value for mar_smoothing
                if len(mar_history) > mar_smoothing:
                    mar_history.pop(0)
                avg_mar = sum(mar_history) / len(mar_history)

                dx, dy = detect_head_movement(landmarks, initial_nose_point)

                current_action = action_sequence[0] if action_sequence else None
                action_completed = False

                if current_action == 'blink':
                    blink_threshold = 0.3
                    consecutive_frames = 3
                    if 'blink_counter' not in locals():
                        blink_counter = 0
                        total_blinks = 0

                    if ear < blink_threshold:#ear falls below 0.3 means you blinked
                        blink_counter += 1
                    else:
                        if blink_counter >= consecutive_frames:
                            total_blinks += 1
                            print(f"Blink detected for action '{current_action.upper()}'.")
                            cv2.putText(frame, f"Blink detected!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            action_completed = True
                            blink_counter = 0
                        else:
                            blink_counter = 0#else no blink detected 

                elif current_action == 'smile':
                    if avg_mar > mar_threshold:
                        cv2.putText(frame, "Smile detected!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print(f"Smile detected for action '{current_action.upper()}'.")
                        action_completed = True
                    else:
                        cv2.putText(frame, "Please smile.", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                elif current_action in ['turn left', 'turn right', 'look up', 'look down']:
                    action_detected = False
                    if current_action == 'turn left' and dx < -30:#horizontal displacements
                        action_detected = True
                    elif current_action == 'turn right' and dx > 30:
                        action_detected = True
                    elif current_action == 'look up' and dy < -20:#vertical displacements
                        action_detected = True
                    elif current_action == 'look down' and dy > 20:
                        action_detected = True

                    if action_detected:
                        cv2.putText(frame, f"{current_action.upper()} detected!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print(f"{current_action.upper()} detected.")
                        action_completed = True
                    else:
                        cv2.putText(frame, f"Please {current_action}.", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                elif current_action == 'follow dot':#follow dot test
                    timeout_duration = 20  

                    if gaze_start_time is None:
                        gaze_start_time = time.time()
                        start_time = gaze_start_time  
                        gaze_index = 0
                        gaze_points = []
                        num_points = 4 #number of points to confirm follow dot 
                        margin = 100  
                        for _ in range(num_points):#random points in frame space
                            x = random.randint(margin, frame_width - margin)
                            y = random.randint(margin, frame_height - margin)
                            gaze_points.append((x, y))
                        gaze_point = gaze_points[gaze_index]
                        point_start_time = time.time()

                    elapsed_time =time.time()-start_time
                    if elapsed_time > timeout_duration:
                        print("Timeout reached for follow dot action.")
                        cv2.putText(frame, "Timeout for follow dot.", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        action_failed=True
                        action_completed = True 
                     
                    else:
                        cv2.circle(frame, gaze_point, 10, (0, 0, 255), -1)

                        left_eye_center = get_eye_center(landmarks, LEFT_EYE)
                        right_eye_center = get_eye_center(landmarks, RIGHT_EYE)
                        eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

                        eye_to_gaze_vector = (gaze_point[0] - eye_center[0], gaze_point[1] - eye_center[1])

                        distance = math.hypot(eye_to_gaze_vector[0], eye_to_gaze_vector[1])

                        gaze_threshold = 100 

                        if distance < gaze_threshold:
                            if time.time() - point_start_time > gaze_movement_time:
                                gaze_index += 1
                                if gaze_index >= len(gaze_points):
                                    cv2.putText(frame, "Gaze following detected!", (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    print("Gaze following detected.")
                                    action_completed = True
                                    gaze_start_time = None
                                else:
                                    gaze_point = gaze_points[gaze_index]
                                    point_start_time = time.time()
                        else:
                            point_start_time = time.time()
                            cv2.putText(frame, "Please follow the dot with your eyes.", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                else:
                     action_completed = False
                       
                if action_completed:
                    if action_failed:
                        print("Action Failed liveness not confirmed")
                        cv2.putText(frame, "Action failed. Liveness not confirmed.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        break
                    else:
                        action_sequence.pop(0)
                        blink_counter = 0
                        total_blinks = 0
                        mar_history.clear()
                        initial_nose_point = landmarks[NOSE_TIP]
                        gaze_start_time = None
                        time.sleep(1)
                        if action_sequence:
                            print(f"Next action: {action_sequence[0].upper()}")

                if not action_sequence and not action_failed:
                    liveness_confirmed = True
                    print("Liveness confirmed based on challenge-response or follow dot tests.")
                    break

            else:
                cv2.putText(frame, "No face detected.", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if action_sequence:
                cv2.putText(frame, f"Current action: {action_sequence[0]}", (10, frame_height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow('Liveness Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
        

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_kyc', methods=['POST'])
def start_kyc():
    global kyc_started
    kyc_started = True
    print("KYC process has started!")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
