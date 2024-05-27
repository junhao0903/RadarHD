#  File for testing RadarHD

import json

import torch

from torchsummary import summary

from train_test_utils.dataloader import *
from train_test_utils.model import *
import cv2

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

"""
## Constants. Edit this to change the model to test on.
"""

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

MIN_THRESHOLD = 1
MAX_THRESHOLD = 255

params = {
    'model_name': '13',
    'expt': 1,
    'dt': '20240524-192545',
    'epoch_num': 120,
    'data': 5,
    'gpu': 1,
}

radar_img_list = []

def dump_tensor(name, data):
    data_cpu = data.cpu()
    tensor_data_array = np.array(data_cpu.numpy()).astype(np.float32)
    tensor_data_array.tofile(os.path.join('/home/adt/deeplearning/pth/radarhd/new/', name + '.bin'))

def dump_pc(name, data):
    tensor_data_array = data.astype(np.float32)
    tensor_data_array.tofile(os.path.join('/home/adt/deeplearning/pth/radarhd/new/', name + '.bin'))

def pcl_to_polar(pcl_data):
    # Filter pcl data based on x y z values
    mask = ((pcl_data[:, 0] > 0) & ((pcl_data[:, 0] <= X_MAX) &
                                    ((pcl_data[:, 2] >= Z_MIN) & ((pcl_data[:, 2] <= Z_MAX) &
                                                                  ((pcl_data[:, 1] >= -Y_MAX) & (
                                                                          pcl_data[:, 1] <= Y_MAX))))))
    pcl_data = pcl_data[mask, :]
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


def create_image_polar(polar_data):
    mask = ((polar_data[:, 0] > 0) & (polar_data[:, 0] <= R_MAX))
    polar_data = polar_data[mask, :]

    r = polar_data[:, 0]
    a = polar_data[:, 1]
    intensity = polar_data[:, 2]

    image = np.zeros((RBINS, ABINS_RADAR))
    r_grid = np.linspace(0, R_MAX, RBINS)
    a_grid = np.linspace(-90, 90, ABINS_RADAR)

    min_intensity = np.min(intensity)
    max_intensity = np.max(intensity)
    intensity = (intensity - min_intensity) / (max_intensity - min_intensity)

    for i in range(polar_data.shape[0]):
        ri = np.argmax(r_grid >= r[i])
        ai = np.argmax(a_grid >= a[i])

        if (image[ri, ai] == 0) | (intensity[i] > image[ri, ai]):
            image[ri, ai] = intensity[i]
    return (image * 255).astype(np.uint8)

def create_radar_tensor():
    X = torch.Tensor([])
    for i, radar_img in enumerate(radar_img_list):
        dump_pc(str(i), np.asarray(radar_img))
        xx = torch.Tensor(np.reshape(np.asarray(radar_img) / 255.0, (1, RBINS, ABINS_RADAR)))
        X = torch.cat((X, xx), dim=0)
    return X.unsqueeze(0).to(device)


def prepare_model():
    global model, history_len, device
    print(torch.__version__)
    torch.manual_seed(0)

    # Can be set to cuda/cpu. Make sure model and data are moved to cuda if cuda is used
    if params['gpu'] == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    name_str = params['model_name'] + '_' + str(params['expt']) + '_' + params['dt']
    LOG_DIR = './logs/' + name_str + '/'
    with open(os.path.join(LOG_DIR, 'params.json'), 'r') as f:
        train_params = json.load(f)

    # Define model
    history_len = train_params['history'] + 1
    model = UNet1(history_len, 1).to(device)
    summary(model, (history_len, 256, 64))

    epoch_num = '%03d' % params['epoch_num']
    model_file = LOG_DIR + epoch_num + '.pt_gen'
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # Testing
    model.eval()


def detect_objects(test_data):
    with torch.no_grad():
        pred = model(test_data)
        pred = np.squeeze(pred.cpu().numpy())
        pred = (pred * 255).astype(np.uint8)
        return pred


def prepare_pol2cart():
    global x_axis, y_axis, x_axis_grid, y_axis_grid
    agrid = np.linspace(-90, 90, ABINS_LIDAR)
    rgrid = np.linspace(0, R_MAX, RBINS)

    cosgrid = np.cos(agrid * np.pi / 180)
    singrid = np.sin(agrid * np.pi / 180)

    sine_theta, range_d = np.meshgrid(singrid, rgrid)
    cos_theta = np.sqrt(1 - sine_theta ** 2)

    x_axis = np.multiply(range_d, cos_theta)
    y_axis = np.multiply(range_d, sine_theta)

    x_axis_grid = np.linspace(0, R_MAX, RBINS)
    y_axis_grid = np.linspace(-R_MAX, R_MAX, ABINS_LIDAR)


def convert_pol2cart(a):
    b = np.zeros((RBINS, ABINS_LIDAR))
    loc = np.argwhere(a > 0)
    xloc = loc[:, 0]
    yloc = loc[:, 1]
    x = x_axis[xloc, yloc]
    y = y_axis[xloc, yloc]
    new_xloc = [np.argmax(x_axis_grid >= x[i]) for i in range(len(x))]
    new_yloc = [np.argmax(y_axis_grid >= y[i]) for i in range(len(y))]
    b[new_xloc, new_yloc] = a[xloc, yloc]
    return b.astype(np.uint8)

def convert2pcd(pred_cart):
    ret, thresh_img = cv2.threshold(pred_cart, MIN_THRESHOLD, MAX_THRESHOLD, cv2.THRESH_TOZERO)

    # 0 dim is azimuth, 1 dim is range
    location = np.squeeze(cv2.findNonZero(thresh_img))

    if location.size == 1:
        point_loc_3d = np.column_stack((np.array([0]), np.array([0]), np.array([0])))
    else:
        y_location = y_axis_grid[location[:, 0]]
        x_location = x_axis_grid[location[:, 1]]
        point_loc_3d = np.column_stack((x_location, y_location, np.zeros(location.shape[0])))
    return point_loc_3d

def read_points(file_path, dim=4, datatype=np.float32):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=datatype).reshape(-1, dim)
    else:
        raise NotImplementedError

def point_cloud_callback():
    # 将点云数据转换为 numpy 数组
    dataset_path = "/home/adt/deeplearning/Data/new_radar/__2024-05-24-19-07-00/radar_bin/"
    files = sorted(glob.glob(dataset_path + '*'))
    files = files[-255:]
    all_radar_files = []
    for id in range(history_len):
        all_radar_files.append(files[id])

    for idx, radar_file in enumerate(all_radar_files):
        radar_data = read_points(radar_file, 8)
        dump_pc("aaa", radar_data)
        polar_radar_data = pcl_to_polar(radar_data)
        dump_pc("bbb", polar_radar_data)
        radar_img = create_image_polar(polar_radar_data)
        dump_pc("ccc", radar_img)
        radar_img_list.append(radar_img)

    if len(radar_img_list) > history_len:
        radar_img_list.pop(0)

    # 对点云进行推理
    if len(radar_img_list) == history_len:
        test_data = create_radar_tensor()
        dump_tensor("input", test_data)
        pred_pol = detect_objects(test_data)
        dump_pc("output", pred_pol)
        pred_cart = convert_pol2cart(pred_pol)
        dump_pc("pred_cart", pred_cart)
        pc = convert2pcd(pred_cart)
        dump_pc("convert2pcd", pc)

def main():
    prepare_model()
    prepare_pol2cart()
    point_cloud_callback()


if __name__ == '__main__':
    main()
