import open3d as o3d
import numpy as np
# import cv2
import matplotlib.pyplot as plt

# def depth2map():



if __name__ == "__main__":
    bag_filename = "data/num1_circular.bag"
    bag_reader = o3d.t.io.RSBagReader()
    bag_reader.open(bag_filename)

    frame = bag_reader.next_frame()
    while not bag_reader.is_eof():
        # process im_rgbd.depth and im_rgbd.color
        frame = bag_reader.next_frame()
        # o3d.t.io.read_image(im_rgbd)
        color = frame.color
        color = np.rot90(np.rot90(color))
        # color_3d = o3d.geometry.Image(color)


        depth = frame.depth
        depth = np.rot90(np.rot90(depth))
        # depth_3d = o3d.geometry.Image(depth)
        # depth_as_img = o3d.geometry.Image((depth).astype(np.uint8))

        # point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_as_img)

        plt.figure(figsize=(30,10))
        plt.subplot(1,2,1)
        plt.imshow(depth)
        plt.subplot(1,2,2)
        plt.imshow(color)
        plt.show()

        # pcd = o3d.io.read_point_cloud(point_cloud.path)
        # print(pcd)
        # print(np.asarray(pcd.points))
        # o3d.visualization.draw_geometries([pcd],
        #                                   zoom=0.3412,
        #                                   front=[0.4257, -0.2125, -0.8795],
        #                                   lookat=[2.6172, 2.0475, 1.532],
                                        #   up=[-0.0694, -0.9768, 0.2024])
        


    bag_reader.close()