#----------------------------------------------------------------------------------#
# By Emanuel Nunez and Edward White
# Version 1.1s
#----------------------------------------------------------------------------------#
# Notes
# No need to change the IP address or port number
# To run the server: 
"""
python3 simple_socket_server.py
"""
#----------------------------------------------------------------------------------#

# Import modules
import socket
import google.generativeai as genai
import cv2 # import cv2 add on
import mediapipe as mp #import mediapipe add on
#AIzaSyDwklaRtjQvow77n8ZgnhoUscRQ2_luZDk
global x
x = "Hello"
genai.configure(api_key="AIzaSyDwklaRtjQvow77n8ZgnhoUscRQ2_luZDk")
model = genai.GenerativeModel("gemini-1.5-flash")

# Define the server (computer) details
host = '0.0.0.0'    # Localhost, always 0.0.0.0, on nao its based on ethernet ip
port = 8888         # Port number

#swing function
class HandTracking:
                def __init__(self):
                    # Initialize MediaPipe Hands
                    self.mp_hands = mp.solutions.hands # Initialize MediaPipe Hands
                    self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.7) # confidence rate for hand detection and tracking # 0.5, 0.5
                    self.mp_draw = mp.solutions.drawing_utils #draw outline on hands for tracking
                    self.current_counter = None #set counter appendices
                    self.sequence = []
                    self.exercise_counter = 0 #exercise counter at 0
                    self.start_counter = 0 #start counter at 0

                def detect_hand_positions(self): # detection for hands
                    video_capture = cv2.VideoCapture(0) #built in webcam 0

                    #ignoring empty camrea frame
                    while video_capture.isOpened():
                        success, image = video_capture.read()
                        if not success:
                            print("Ignoring empty camera frame.")
                            continue

                        # image processing
                        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) #convert BGR to RGB

                        # extract image dimensions
                        h, w, _ = image.shape  # Height, width, and channels

                        # hand detection
                        mediapipe_results = self.hands.process(image)

                        # initialize the variable for index_finger_tip in case no hand is detected
                        index_finger_tips = {"Left": None, "Right": None}  # Track index finger tip for both hands

                        # check results
                        if mediapipe_results.multi_hand_landmarks and mediapipe_results.multi_handedness:
                            for hand_landmarks, hand_label in zip(mediapipe_results.multi_hand_landmarks, mediapipe_results.multi_handedness):

                                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                                # determine hand label (Left or Right)
                                label = hand_label.classification[0].label  # "Left" or "Right"

                                # extract index finger tip position
                                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                                # store the position of the index finger for each hand
                                index_finger_tips[label] = (index_finger_tip.x, index_finger_tip.y)

                        # add transparent rectangles to the corners
                        image = self.add_dynamic_rectangles(image, index_finger_tips, w, h, video_capture)

                        # display video output "Swing Hands Exercise"
                        cv2.imshow('Swing Hands Exercise', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(5) & 0xFF == 27: #ESC key to exit
                            break

                    # Release video capture
                    video_capture.release()
                    cv2.destroyAllWindows()

                def add_dynamic_rectangles(self, image, index_finger_tips, w, h, video_capture):  # define rectangles for image
                    # Create a semi-transparent overlay
                    overlay = image.copy()

                    # Define the transparency and color
                    alpha = 0.3  # Transparency factor
                    active_color = (0, 0, 255)  # Red color (when active)
                    inactive_color = (0, 255, 0)  # Green color (when inactive)
                    blue_color = (255, 0, 0)  # Blue color for the "Triangle stretch" is active

                    # Define the size of the rectangles (quarter size of the image)
                    rect_width = w // 2  # width of rectangle x axis
                    rect_height = h // 2  # height of rectangle y axis

                    # Define the positions of the rectangles
                    rectangles = [
                        (-50, 100),  # Top-left corner
                        (350, 100),  # Top-right corner
                        (-50, 375),  # Bottom-left corner
                        (350, 375)  # Bottom-right corner
                    ]
                    # Track which boxes should be highlighted (red)
                    highlighted_boxes = [False, False, False, False]

                    # Check if either the left or right index finger is inside each rectangle
                    for i, (rect_x, rect_y) in enumerate(rectangles):
                        for hand, (finger_pos) in index_finger_tips.items():
                            if finger_pos:  # Ensure the finger position is valid (not None)
                                finger_x, finger_y = finger_pos
                                if rect_x <= finger_x * w <= rect_x + rect_width and rect_y <= finger_y * h <= rect_y + rect_height:
                                    highlighted_boxes[i] = True  # Mark this box as active
                                    break  # No need to check the other hand for the same box

                    if highlighted_boxes[0] and highlighted_boxes[1]:  # bottom right and left boxes
                        self.state_swing_counter('End', image, w, h, video_capture)  # end of hand movement

                    if highlighted_boxes[2] and highlighted_boxes[3]:  # top right and left
                        self.state_swing_counter('Start', image, w, h, video_capture)  # start of hand movement

                    # Draw the rectangles with the appropriate colour
                    for i, (rect_x, rect_y) in enumerate(rectangles):
                        if highlighted_boxes[i]:
                            color = blue_color if (i in [0, 1, 2, 3]) else active_color  # Set the colour to blue for relevant boxes
                        else:
                            color = inactive_color  # inactive colour green
                        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), color,
                                    -1)  # add rectangles to cv image

                    # Blend the overlay with the image
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)  # blend the overlay with the image

                    return image

                def state_swing_counter(self, counter, image, w, h, video_capture):  # define counter for swing hand detection
                    if self.current_counter != counter:  # if current counter does not equal counter
                        self.current_counter = counter  # make current counter = counter
                        self.sequence.append(counter)  # append counter
                        print(f'Counter updated to: {counter}')  # print what state the counter is

                        # Debugging: Print the current sequence
                        print(f'Current sequence: {self.sequence}')

                        # Check if 'start' is detected and count the occurrences
                        if counter == "Start":
                            self.start_counter += 1
                            print(f"'Start' detected {self.start_counter} times.")

                            # Check if 'Start' has been detected 6 times
                            if self.start_counter >= 6:
                                print("Completed Swing Hand Exercise")  # Print "Complete" when 'Start' is detected 6 times
                                cv2.putText(image, "Complete", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                                            2)
                                video_capture.release()  # Release the camera
                                cv2.destroyAllWindows()  # Close all windows
                                return  # Exit the method and stop the camera

#YOGA function                            
class HandTracking2:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_tracking_confidence=0.5, min_detection_confidence=0.5) # confidence rate for hand detection and tracking
        self.mp_draw = mp.solutions.drawing_utils
        self.triangle_stretch_counter = 0 # Counter for triangle stretches
        print("1.Stretch to the Top left corner with your left hand, and with your right hand stretch to the bottom right corner") #instructions for the user to follow
        print("2.Stretch to the Bottom left corner with your left hand, and with your right hand stretch to the top right corner") #instructions for the user to follow

    def detect_hand_positions2(self): # detection for hands
        video_capture = cv2.VideoCapture(0) # Open webcam

        # ignoring empty camrea frame
        while video_capture.isOpened():
            # Step 1: Read frame from camera
            success, image = video_capture.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # image processing
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # extract image dimensions
            h, w, _ = image.shape  # Height, width, and channels

            # hand detection
            mediapipe_results = self.hands.process(image)

            # initialize the variable for index_finger_tip in case no hand is detected
            index_finger_tips = {"Left": None, "Right": None}  # track index finger tip for both hands

            #check results
            if mediapipe_results.multi_hand_landmarks and mediapipe_results.multi_handedness:
                for hand_landmarks, hand_label in zip(
                    mediapipe_results.multi_hand_landmarks, mediapipe_results.multi_handedness
                ):
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Determine hand label (Left or Right)
                    label = hand_label.classification[0].label  # "Left" or "Right"

                    # Extract index finger tip position
                    index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Store the position of the index finger for each hand
                    index_finger_tips[label] = (index_finger_tip.x, index_finger_tip.y)

            # draw rectangles and check for completed stretches
            image = self.draw_rectangles_and_check(image, index_finger_tips, w, h)

            # display the video feed with rectangles
            cv2.imshow('Hand Tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        # release video capture
        video_capture.release()
        cv2.destroyAllWindows()
        

    def draw_rectangles_and_check(self, image, index_finger_tips, w, h):
        #create a semi-transparent overlay
        overlay = image.copy()

        # define the size of the rectangles (quarter size of the image)
        rect_width = w // 4
        rect_height = h // 4

        # define the positions of the rectangles
        rectangles = [
            (0, 0),  # Top-left corner
            (w - rect_width, 0),  # Top-right corner
            (0, h - rect_height),  # Bottom-left corner
            (w - rect_width, h - rect_height)  # Bottom-right corner
        ]

        # Colors for inactive and active states
        inactive_color = (0, 255, 0)  # Green color for inactive
        active_color = (0, 0, 255)  # Red color for active

        # Track which boxes are highlighted
        highlighted_boxes = [False, False, False, False]

        # Check if either the left or right index finger is inside each rectangle
        for i, (rect_x, rect_y) in enumerate(rectangles):
            for hand, finger_pos in index_finger_tips.items():
                if finger_pos:  # Ensure the finger position is valid (not None)
                    finger_x, finger_y = finger_pos
                    if rect_x <= finger_x * w <= rect_x + rect_width and rect_y <= finger_y * h <= rect_y + rect_height:
                        highlighted_boxes[i] = True
                        break

        # Draw the rectangles on the overlay
        for i, (rect_x, rect_y) in enumerate(rectangles):
            color = active_color if highlighted_boxes[i] else inactive_color
            cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), color, -1)

        # Check completion of triangle stretch sequences
        if highlighted_boxes[0] and highlighted_boxes[3]:  # Top-left and Bottom-right
            if self.triangle_stretch_counter == 0:
                self.triangle_stretch_counter += 1
                print(f"Triangle stretch {self.triangle_stretch_counter} completed.")

        if highlighted_boxes[1] and highlighted_boxes[2]:  # Top-right and Bottom-left
            if self.triangle_stretch_counter == 1:
                self.triangle_stretch_counter += 1
                print(f"Triangle stretch {self.triangle_stretch_counter} completed.")
                print("Well done!")
                exit(0)  # Exit the program after completion

        # Blend the overlay with the original image
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        return image


class SimpleServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None

    def start_server(self):
        
        ### 1. Create a socket object
        # Create a socket object
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind the socket to the address and port
        self.server_socket.bind((self.host, self.port))
        # Listen for incoming connections
        self.server_socket.listen(1)

        ### 2. Wait for a connection (forever)
        print("Waiting for incoming connection...")

        try:
            while True:
                #---SERVER STUFF: If a connection is made, accept it---#
                client_socket, client_address = self.server_socket.accept()
                print(f"Connection from {client_address} has been established.")

                #---SERVER STUFF: Handle client communication---#
                self.handle_client(client_socket) ## Thats where the function is being handled

        except KeyboardInterrupt:
            print("Server has been closed.")
        finally:
            self.server_socket.close()

    ## Thats code example
    """
    def handle_client(self, client_socket):
        ### 3. Receive the data from the client
        important_message = client_socket.recv(1024).decode()
        print(f"Received the message: {important_message}")
        
        # If we get the right answer, do complex maths to answer the client's request
        if important_message == "1":
            print("Sending a MESSAGE back to the client")
            important_answer = "THE SERVER WORKS"
        
            #---SERVER STUFF: Sends the output to the client---#
            client_socket.sendall(str(important_answer).encode())
        
        if important_message == "2":
            print("Sending a second MESSAGE back to the client")
            important_answer = "your mom gay"
        
            #---SERVER STUFF: Sends the output to the client---#
            client_socket.sendall(str(important_answer).encode())

        # If we get the wrong answer...
        else:
            important_answer = "Huh?"
            #---SERVER STUFF: Sends the output to the client---#
            client_socket.sendall(str(important_answer).encode())

        #---SERVER STUFF: Close the client socket---#
        client_socket.close()
    """  
    # Attempt 1 code AI integration
    def handle_client(self, client_socket):
        # Receive the data from the client
        important_message = client_socket.recv(1024).decode()
        print(f"Received the message: {important_message}")
        
        # If we get the right answer
        if important_message != "Swing":
            if important_message != "Tai":
                print("AISending a MESSAGE back to the client")
                response = model.generate_content(str(important_message)+"in 20 words")
                print(response.text)
                important_answer = str(response.text)
                #---SERVER STUFF: Sends the output to the client---#
                client_socket.sendall(str(important_answer).encode())
            
        if(important_message == "Tai"): 
            print("Gesture start")
            """
            important_answer = "Now you try doing it"      
            #---SERVER STUFF: Sends the output to the client---#
            client_socket.sendall(str(important_answer).encode())
            """
            tracker = HandTracking2()
            tracker.detect_hand_positions2()
            important_answer = "Well Done on finishin the Ping activity"
            client_socket.sendall(str(important_answer).encode())
        
        if important_message == "Swing":
            print("Swing Gesture start")
            """
            important_answer = "Now you try doing it"      
            #---SERVER STUFF: Sends the output to the client---#
            client_socket.sendall(str(important_answer).encode())
            """
            tracker = HandTracking()
            tracker.detect_hand_positions()
            important_answer = "Well done on finishing the Swing activity"
            #important_message = client_socket.recv(1024).decode()
            client_socket.sendall(str(important_answer).encode())
        
        """
        # If we get the wrong answer...
        else:
            important_answer = "Huh?"
            #---SERVER STUFF: Sends the output to the client---#
            client_socket.sendall(str(important_answer).encode())
        """
        #---SERVER STUFF: Close the client socket---#
        client_socket.close()
    

if __name__ == "__main__":
    server = SimpleServer(host, port)
    server.start_server()
