''' bash script to run the code
pip install opencv-python
pip install tqdm
python3 video_frame_sampler.py
'''

# Imports
import cv2
import os
import json
import base64
from tqdm import tqdm
import shutil

# Paths
input_folder = 'example_data/videos/ego4d'  # Folder containing the mp4 files
output_folder = 'example_data/frames'  # Folder where the jpg frames will be saved
json_output_path = 'example_data/frames_train.json'  # Path to save the JSON file

# Ensure the output folder exists and clean it
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Remove all contents in the output folder
os.makedirs(output_folder, exist_ok=True)

# Clean the JSON file if it exists
if os.path.exists(json_output_path):
    os.remove(json_output_path)

# Frame sampling parameters
frames_to_sample = 5  # Number of frames
sample_interval = 10  # Interval in seconds to sample frames (every 10th second)
frame_skip = 2  # Number of frames to skip between samples
frame_batch = 0  # Video counter
# num_test_vids = 4 # Test counter

# Initialize the list to hold JSON objects
json_list = []

# Get the list of mp4 files
video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

# Iterate over all mp4 files with a progress bar
for file_name in tqdm(video_files, desc="Processing Videos"):
    video_path = os.path.join(input_folder, file_name)
    
    print(f"Processing video: {video_path}")
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
    
    # Calculate the frame numbers to capture
    group_number = 0

    while True:
        # Calculate the frame number for the start of each 10th second
        start_frame_number = int(group_number * sample_interval * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        
        # List to store image file names for the current group
        image_files = []

        # Capture frames_to_sample frames, skipping every frame_skip frames between each capture
        for i in range(frames_to_sample):
            success, frame = video.read()
            if not success:
                break

            # Save the current frame
            output_file_name = f"{frame_batch:04d}-{i+1}.jpg"
            output_file_path = os.path.join(output_folder, output_file_name)
            cv2.imwrite(output_file_path, frame)
            
            # Add the image file name to the list
            image_files.append(output_file_name)

            # Skip the next 2 frames (move to the 3rd frame)
            for _ in range(frame_skip):
                if not video.grab():
                    break
        
        if len(image_files) == frames_to_sample:
            next_image_file_path = os.path.join(output_folder, image_files[frames_to_sample - 1])
            next_image = cv2.imread(next_image_file_path)
            _, buffer = cv2.imencode('.jpg', next_image) # Convert the image to a byte array
            next_image_base64 = base64.b64encode(buffer).decode('utf-8') # Encode the byte array to base64
            
            # Only add JSON object to the group if it's complete and not in our eval
            if not (frame_batch == 439 or frame_batch == 329):
                json_object = {
                    "system_prompt": "Assistant predicts the next image in the image sequence.",
                    "image": image_files[:frames_to_sample - 1],
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Help me generate the next image in this logical sequence of images.<image><image><image><image>"
                        },
                        {
                            "from": "gpt",
                            "value": f"Here is the next image in base64: {next_image_base64}"
                        }
                    ]
                }

                # Add the JSON object to the list
                json_list.append(json_object)
            
            frame_batch += 1
            group_number += 1
        else:
            # Break if the group is incomplete
            break

    video.release()
    
    # Test with only num_test_vids video first
    # num_test_vids -= 1
    # if num_test_vids == 0:
    #     break

# Save the list of JSON objects to a file
with open(json_output_path, 'w') as json_file:
    json.dump(json_list, json_file, indent=4)

print(f"Frame extraction and JSON creation completed. JSON saved to {json_output_path}.")
