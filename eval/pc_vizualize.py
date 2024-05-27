import os
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

root_folder = './processed_imgs_13_1_20240524-192545_test_imgs/'
epoch = '120'

for traj in os.listdir(root_folder):
    if traj.startswith('.'):
        continue
    pred_folder = os.path.join(root_folder, traj, epoch, 'pred', 'pcd')
    label_folder = os.path.join(root_folder, traj, epoch, 'label', 'pcd')

    pred_file_names = sorted(os.listdir(pred_folder), key=lambda x: int(x.split('_')[2]))
    label_file_names = sorted(os.listdir(label_folder), key=lambda x: int(x.split('_')[2]))

    fig = plt.figure(figsize=(10, 5))

    for label_file, pred_file in zip(label_file_names, pred_file_names):
        label = o3d.io.read_point_cloud(os.path.join(label_folder, label_file))
        pred = o3d.io.read_point_cloud(os.path.join(pred_folder, pred_file))
        label = np.asarray(label.points)
        pred = np.asarray(pred.points)

        ax1 = fig.add_subplot(121)
        ax1.scatter(label[:, 1], label[:, 0])
        ax1.set_title(f'Traj No {traj} Label')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim([-50, 50])
        ax1.set_ylim([0, 50])
        ax1.grid(True)

        ax2 = fig.add_subplot(122)
        ax2.scatter(pred[:, 1], pred[:, 0])
        ax2.set_title(f'Traj No {traj} RadarHD')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlim([-50, 50])
        ax2.set_ylim([0, 50])
        ax2.grid(True)

        plt.pause(0.01)

plt.show()
