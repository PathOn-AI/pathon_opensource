import cv2
import os
import glob
from pathlib import Path

def create_video_from_images(image_folder, output_video, start_frame=515, end_frame=749, fps=30):
    """
    Create a video from a sequence of images.
    
    Args:
        image_folder (str): Path to folder containing images
        output_video (str): Output video filename
        start_frame (int): Starting frame number
        end_frame (int): Ending frame number
        fps (int): Frames per second for output video
    """
    
    # Get list of image files in the specified range
    image_files = []
    for i in range(start_frame, end_frame + 1):
        image_path = os.path.join(image_folder, f"image_on_robot_{i}.jpg")
        if os.path.exists(image_path):
            image_files.append(image_path)
        else:
            print(f"Warning: Image {image_path} not found, skipping...")
    
    if not image_files:
        print(f"No images found in range {start_frame}-{end_frame}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Error: Could not read first image {image_files[0]}")
        return
    
    height, width, layers = first_image.shape
    print(f"Video dimensions: {width}x{height}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        return
    
    # Process each image
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        if img is not None:
            # Resize image if needed (ensure all images have same dimensions)
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            video_writer.write(img)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
        else:
            print(f"Warning: Could not read image {image_file}")
    
    # Release everything
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video created successfully: {output_video}")
    print(f"Total frames: {len(image_files)}")
    print(f"Duration: {len(image_files)/fps:.2f} seconds")

def main():
    # Configuration
    IMAGE_FOLDER = "images"  # Adjust this path to your images folder
    OUTPUT_VIDEO = "robot_sequence_515_749_on_robot.mp4"
    START_FRAME = 515
    END_FRAME = 749
    FPS = 30  # Adjust frame rate as needed
    
    # Check if image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder '{IMAGE_FOLDER}' not found")
        print("Please update the IMAGE_FOLDER path in the script")
        return
    
    print(f"Creating video from frames {START_FRAME} to {END_FRAME}")
    print(f"Input folder: {IMAGE_FOLDER}")
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"FPS: {FPS}")
    print("-" * 50)
    
    create_video_from_images(
        image_folder=IMAGE_FOLDER,
        output_video=OUTPUT_VIDEO,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        fps=FPS
    )

if __name__ == "__main__":
    main()