import os
import numpy as np
import pickle
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import argparse
import glob as glob

# Parameters
X_MAX = 40
Y_MAX = 40
Z_MIN = -1
Z_MAX = 3

R_MAX = 50
RBINS = 256
ABINS_LIDAR = 512
ABINS_RADAR = 64

RANGE_FFT = 512
AZIM_FFT = 64
MAX_RANGE = 21.59

# ########################################
def mkdir(path: str):
    if not os.path.exists(path):
        print("新建目录<{}>".format(path))
        os.makedirs(path, exist_ok=True)


def read_points(file_path, dim=4, datatype=np.float32):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=datatype).reshape(-1, dim)
    else:
        raise NotImplementedError


def main(args):
    # 数据集按照比例划分，多余的给test集
    assert len(args.split) == 3
    assert os.path.exists(args.save_path)
    assert os.path.exists(args.data_path)
    datasets = os.listdir(args.data_path)
    datasets.sort(key=lambda x: x)
    for id, dataset in enumerate(datasets):
        if "__" == dataset[:2]:  # 数据集以"__"开头
            dataset_path = os.path.join(args.data_path, dataset)
            print(dataset_path)
            get_lidar_pcl(id, dataset_path, args)
            get_radar_pcl(id, dataset_path, args)


def pcl_to_polar(pcl_data, is_lidar):
    # Filter pcl data based on x y z values
    mask = ((pcl_data[:, 0] > 0) & ((pcl_data[:, 0] <= X_MAX) &
                                    ((pcl_data[:, 2] >= Z_MIN) & ((pcl_data[:, 2] <= Z_MAX) &
                                                                  ((pcl_data[:, 1] >= -Y_MAX) & (
                                                                          pcl_data[:, 1] <= Y_MAX))))))
    pcl_data = pcl_data[mask, :]

    # if not is_lidar:
    #     mask = (pcl_data[:, 5] == 4)
    #     pcl_data = pcl_data[mask, :]

    polar_data = np.zeros((pcl_data.shape))

    for i in range(pcl_data.shape[0]):
        xi = pcl_data[i, 0]
        yi = pcl_data[i, 1]
        zi = pcl_data[i, 2]
        ri = np.sqrt(xi ** 2 + yi ** 2 + zi ** 2)
        ai = np.rad2deg(np.arctan2(yi, xi))
        ei = np.rad2deg(np.arcsin(zi / ri))

        polar_data[i, 0] = ri
        polar_data[i, 1] = ai
        polar_data[i, 2] = ei

    return polar_data


def create_image_polar(polar_data, is_lidar, is_binary):
    mask = ((polar_data[:, 0] > 0) & (polar_data[:, 0] <= R_MAX))
    polar_data = polar_data[mask, :]

    r = polar_data[:, 0]
    a = polar_data[:, 1]
    intensity = polar_data[:, 2]

    if is_lidar:
        image = np.zeros((RBINS, ABINS_LIDAR))
        r_grid = np.linspace(0, R_MAX, RBINS)
        a_grid = np.linspace(-90, 90, ABINS_LIDAR)

    else:
        image = np.zeros((RBINS, ABINS_RADAR))
        r_grid = np.linspace(0, R_MAX, RBINS)
        a_grid = np.linspace(-90, 90, ABINS_RADAR)

    min_intensity = np.min(intensity)
    max_intensity = np.max(intensity)
    intensity = (intensity - min_intensity) / (max_intensity - min_intensity)

    for i in range(polar_data.shape[0]):
        ri = np.argmax(r_grid >= r[i])
        ai = np.argmax(a_grid >= a[i])

        if (image[ri, ai] == 0) | ((intensity[i] > image[ri, ai]) & is_binary == False):
            if is_binary:
                image[ri, ai] = 1
            else:
                image[ri, ai] = intensity[i]

    if is_binary:
        image = image.astype(np.bool_)

    return image


def test_pcd(idx, lidar_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_data[:, :3])
    o3d.io.write_point_cloud("/home/adt/z1/" + str(idx) + ".pcd", pcd, write_ascii=True)  # save xyz.pcd
    mask = ((lidar_data[:, 2] > -0.3) & (lidar_data[:, 0] < 25))
    lidar_data = lidar_data[mask, :]
    pcd.points = o3d.utility.Vector3dVector(lidar_data[:, :3])
    o3d.io.write_point_cloud("/home/adt/z2/" + str(idx) + ".pcd", pcd, write_ascii=True)  # save xyz.pcd


def get_lidar_pcl(id, dataset_path, args):
    train_ratio = args.split[0]
    test_ratio = args.split[1]
    val_ratio = args.split[2]
    total_size = len(glob.glob(os.path.join(dataset_path, "lidar_bin", "*")))
    train_size = int(total_size * train_ratio / (train_ratio + test_ratio + val_ratio))
    val_size = int(total_size * val_ratio / (train_ratio + test_ratio + val_ratio))
    test_size = total_size - train_size - val_size

    all_lidar_files = sorted(glob.glob(os.path.join(dataset_path, "lidar_bin", "*")))
    lidar_image = np.zeros((len(all_lidar_files), RBINS, ABINS_LIDAR))

    for idx, lidar_file in tqdm(enumerate(all_lidar_files), total=len(all_lidar_files)):
        lidar_data = read_points(os.path.join(dataset_path, lidar_file), 4)
        polar_lidar_data = pcl_to_polar(lidar_data, is_lidar=True)
        lidar_img = create_image_polar(polar_lidar_data, is_lidar=True, is_binary=False)
        lidar_image[idx, :, :] = lidar_img

        file_name = 'L_' + str(id) + '_' + str(idx) + '.png'
        if idx <= train_size:
            curr_folder = os.path.join(args.save_path, "train", "lidar")
        elif idx > train_size and idx <= val_size + train_size:
            curr_folder = os.path.join(args.save_path, "val", "lidar")
        else:
            curr_folder = os.path.join(args.save_path, "test", "lidar")
        mkdir(curr_folder)
        file_name = os.path.join(curr_folder, file_name)

        im = Image.fromarray((lidar_img * 255).astype(np.uint8))
        im.save(file_name)

    return lidar_image


def get_radar_pcl(id, dataset_path, args):
    train_ratio = args.split[0]
    test_ratio = args.split[1]
    val_ratio = args.split[2]
    total_size = len(glob.glob(os.path.join(dataset_path, "radar_bin", "*")))
    train_size = int(total_size * train_ratio / (train_ratio + test_ratio + val_ratio))
    val_size = int(total_size * val_ratio / (train_ratio + test_ratio + val_ratio))
    test_size = total_size - train_size - val_size

    all_radar_files = sorted(glob.glob(os.path.join(dataset_path, "radar_bin", "*")))
    radar_image = np.zeros((len(all_radar_files), RBINS, ABINS_RADAR))

    for idx, radar_file in tqdm(enumerate(all_radar_files), total=len(all_radar_files)):
        radar_data = read_points(os.path.join(dataset_path, radar_file), 8)
        polar_radar_data = pcl_to_polar(radar_data, is_lidar=False)
        radar_img = create_image_polar(polar_radar_data, is_lidar=False, is_binary=False)
        radar_image[idx, :, :] = radar_img

        file_name = 'R_' + str(id) + '_' + str(idx) + '.png'
        if idx <= train_size:
            curr_folder = os.path.join(args.save_path, "train", "radar")
        elif idx > train_size and idx <= val_size + train_size:
            curr_folder = os.path.join(args.save_path, "val", "radar")
        else:
            curr_folder = os.path.join(args.save_path, "test", "radar")
        mkdir(curr_folder)
        file_name = os.path.join(curr_folder, file_name)

        im = Image.fromarray((radar_img * 255).astype(np.uint8))
        im.save(file_name)

    return radar_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data-path', default='/files/data/test', help='your data root path')
    parser.add_argument('--save-path', default='/files/data/test', help='your data save path')
    parser.add_argument('--split', type=list, default=[18, 1, 1], help='train_size: test_size: val_size')
    args = parser.parse_args()

    main(args)
