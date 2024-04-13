#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
import pdb
import matplotlib.pyplot as plt

  
def estimate_coef(x, y):
  # number of observations/points
  n = np.size(x)
 
  # mean of x and y vector
  m_x = np.mean(x)
  m_y = np.mean(y)
 
  # calculating cross-deviation and deviation about x
  SS_xy = np.sum(y*x) - n*m_y*m_x
  SS_xx = np.sum(x*x) - n*m_x*m_x
 
  # calculating regression coefficients
  b_1 = SS_xy / SS_xx
  b_0 = m_y - b_1*m_x
 
  return (b_0, b_1)

   
def plot_regression_line(x, y, b):
  # plotting the actual points as scatter plot
  plt.scatter(x, y, color = "m",
        marker = "o", s = 30)
 
  # predicted response vector
  y_pred = b[0] + b[1]*x
 
  # plotting the regression line
  plt.plot(x, y_pred, color = "g")
 
  # putting labels
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()

def detect_cater(img):
    height, width = img.shape
    x_medi = np.median(img, axis=1)
    y_medi = np.median(img, axis=0)
    x_mask = np.ones((height,width))
    y_mask = np.zeros((height,width))

    for h in range(height):
        x_array = img[h, :]
        x_array = x_array[x_array != 0]
        outline = np.where((x_array >= x_medi[h] - 200) & (x_array <= x_medi[h] + 200))[0]
        x_mask[h, outline] = 0
    # plt.imshow(x_mask)
    # plt.show()

    for w in range(width):
        # deleting zero depth
        y_array = img[:, w]
        x_array = np.where(y_array != 0)[0]
        y_array = y_array[y_array != 0]
        if (len(y_array) < 2):
            continue
        
        # RANSAC
        ransac = RANSACRegressor(random_state=0).fit(x_array.reshape(-1,1), y_array)
        # y_prad = ransac.predict(x_range)
        # ransac_coef = ransac.estimator_.coef_
        outlier = np.where(~ransac.inlier_mask_)[0]
        # pdb.set_trace()
        y_mask[outlier, w] = 1

        # plt.scatter(x_array[inlier_mask], y_array[inlier_mask], color="blue", label="Inliers")
        # plt.scatter(x_array[outlier_mask], y_array[outlier_mask], color="red", label="Outliers")
        # plt.title("RANSAC - outliers vs inliers")
        # plt.show()

    plt.figure(figsize=(30,30))
    plt.subplot(1,3,1)
    plt.imshow(x_mask)
    plt.subplot(1,3,2)
    plt.imshow(y_mask)
    plt.subplot(1,3,3)
    plt.imshow(np.logical_and(y_mask, x_mask))
    plt.show()
    return outline



# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()
try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    
    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        depth_color_image = depth_color_image[:240, :]
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image[:240, :]
        # color_image = np.asanyarray(color_frame.get_data())

        # Render image in opencv window
        depth_color_image = cv2.rotate(depth_color_image, cv2.ROTATE_180)
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)

        # color_image = cv2.rotate(color_image, cv2.ROTATE_180)
        cv2.imshow("Depth Stream", depth_color_image)
        # cv2.imshow("color stream", color_frame)


        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            mask = detect_cater(depth_image)
            cv2.destroyAllWindows()
            break

finally:
    pass

