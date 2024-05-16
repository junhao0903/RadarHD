import os
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from tqdm import tqdm

def pc_distance(pc_A, pc_B, type, bin_size):
    pc_A = bin_pc(pc_A, bin_size)
    pc_B = bin_pc(pc_B, bin_size)

    if type == "chamfer":
        distanceA = 0
        for i in range(pc_A.shape[0]):
            distanceA += np.min(np.sqrt(np.sum((pc_B - pc_A[i]) ** 2, axis=1)))
        distanceA /= pc_A.shape[0]

        distanceB = 0
        for i in range(pc_B.shape[0]):
            distanceB += np.min(np.sqrt(np.sum((pc_A - pc_B[i]) ** 2, axis=1)))
        distanceB /= pc_B.shape[0]

        distance = 0.5 * distanceA + 0.5 * distanceB

    elif type == "hausdorff":
        distanceA = np.zeros((pc_A.shape[0], 1))
        for i in range(pc_A.shape[0]):
            distanceA[i] = np.min(np.sqrt(np.sum((pc_B - pc_A[i]) ** 2, axis=1)))

        distanceB = np.zeros((pc_B.shape[0], 1))
        for i in range(pc_B.shape[0]):
            distanceB[i] = np.min(np.sqrt(np.sum((pc_A - pc_B[i]) ** 2, axis=1)))
        distance = max([np.max(distanceA), np.max(distanceB)])

    elif type == "mod_hausdorff":
        distanceA = np.zeros((pc_A.shape[0], 1))
        for i in range(pc_A.shape[0]):
            distanceA[i] = np.min(np.sqrt(np.sum((pc_B - pc_A[i]) ** 2, axis=1)))

        distanceB = np.zeros((pc_B.shape[0], 1))
        for i in range(pc_B.shape[0]):
            distanceB[i] = np.min(np.sqrt(np.sum((pc_A - pc_B[i]) ** 2, axis=1)))
        distance = max([np.median(distanceA), np.median(distanceB)])

    elif type == "l1":
        RMAX = 10.8
        x_axis_new_grid = np.arange(0, RMAX + bin_size, bin_size)
        y_axis_new_grid = np.arange(-RMAX, RMAX + bin_size, bin_size)
        map_A = np.zeros((len(x_axis_new_grid), len(y_axis_new_grid)))
        map_B = np.zeros((len(x_axis_new_grid), len(y_axis_new_grid)))
        for i in range(pc_A.shape[0]):
            x_loc = np.argmax(x_axis_new_grid >= pc_A[i, 0])
            y_loc = np.argmax(y_axis_new_grid >= pc_A[i, 1])
            map_A[x_loc, y_loc] = 1
        for i in range(pc_B.shape[0]):
            x_loc = np.argmax(x_axis_new_grid >= pc_B[i, 0])
            y_loc = np.argmax(y_axis_new_grid >= pc_B[i, 1])
            map_B[x_loc, y_loc] = 1
        distance = np.sum(np.abs(map_A - map_B))

    return distance


def bin_pc(pc, bin_size):
    if bin_size == 0:
        new_pc = pc
    else:
        RMAX = 10.8
        x_axis_new_grid = np.arange(0, RMAX + bin_size, bin_size)
        y_axis_new_grid = np.arange(-RMAX, RMAX + bin_size, bin_size)
        new_pc = np.zeros((pc.shape[0], 3))
        for i in range(pc.shape[0]):
            new_pc[i, 0] = x_axis_new_grid[np.argmax(x_axis_new_grid >= pc[i, 0])]
            new_pc[i, 1] = y_axis_new_grid[np.argmax(y_axis_new_grid >= pc[i, 1])]
    return new_pc

def reorder_dir(folder):
    file_names = os.listdir(folder)
    file_names = sorted(file_names, key=lambda x: int(x.split('_')[2]))
    return file_names

root_folder = './processed_imgs_13_1_20220320-034822_test_imgs'

trajs = os.listdir(root_folder)
epoch = '120'
chamfer_all_distance = [None] * len(trajs)
hausdorff_all_distance = [None] * len(trajs)
mod_hausdorff_all_distance = [None] * len(trajs)

bin_size = 0

# Choose the index of the trajectories in 'trajs' you want to test
which_traj = range(2, len(trajs))

a = []
c = []

for k in which_traj:
    traj_name = trajs[k]
    print(traj_name)

    pred_folder = os.path.join(root_folder, traj_name, epoch, 'pred', 'pcd')
    pred_file_names = reorder_dir(pred_folder)

    label_folder = os.path.join(root_folder, traj_name, epoch, 'label', 'pcd')
    label_file_names = reorder_dir(label_folder)

    chamfer_dist = np.zeros(len(label_file_names))
    mod_hausdorff_dist = np.zeros(len(label_file_names))

    for j, label_file_name in tqdm(enumerate(label_file_names), total=len(label_file_names)):
        label = o3d.io.read_point_cloud(os.path.join(label_folder, label_file_name))
        pred = o3d.io.read_point_cloud(os.path.join(pred_folder, pred_file_names[j]))
        label = np.asarray(label.points)
        pred = np.asarray(pred.points)

        chamfer_dist[j] = pc_distance(label[:, :2], pred[:, :2], "chamfer", bin_size)
        mod_hausdorff_dist[j] = pc_distance(label[:, :2], pred[:, :2], "mod_hausdorff", bin_size)

    chamfer_all_distance[k] = chamfer_dist
    mod_hausdorff_all_distance[k] = mod_hausdorff_dist

    a.extend(chamfer_all_distance[k])
    c.extend(mod_hausdorff_all_distance[k])

plt.figure()
h = plt.hist(a, bins=100, density=True, cumulative=True, color='red', histtype='step', linewidth=2)
h = plt.hist(c, bins=100, density=True, cumulative=True, linestyle='--', color='red', histtype='step', linewidth=2)

plt.legend(['Chamfer (Ours against Lidar)', 'Mod Hausdorff (Ours against Lidar)'])
plt.xlabel('Point Cloud Error (in meters)')
plt.ylabel('CDF')
plt.savefig('pc_compare.png')  # 将图像保存为 plot.png 文件
plt.show()
