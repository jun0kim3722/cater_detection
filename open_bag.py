import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

bag_filename = "data/num1_circular.bag"
# o3d.t.io.RealSenseSensor.list_devices()
bag_reader = o3d.t.io.RSBagReader()
bag_reader.open(bag_filename)
# vis = o3d.visualization.Visualizer()
# # vis.create_window()

frame = bag_reader.next_frame()
while not bag_reader.is_eof():
    # process im_rgbd.depth and im_rgbd.color
    frame = bag_reader.next_frame()
    # o3d.t.io.read_image(im_rgbd)
    depth = frame.depth
    depth_array = np.array(depth)
    img_rotate_180 = cv2.rotate(depth_array, cv2.ROTATE_180)
    cv2.imshow(depth_array)
    # plt.imshow(depth_array), plt.show()
    # o3d.visualization.draw_geometries([depth])

# vis.destroy_window()
bag_reader.close()