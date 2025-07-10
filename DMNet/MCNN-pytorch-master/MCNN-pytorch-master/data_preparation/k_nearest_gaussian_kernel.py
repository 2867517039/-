# import numpy as np
# import scipy
# import scipy.io as io
# from scipy.ndimage import gaussian_filter
# import os
# import glob
# from matplotlib import pyplot as plt
# import h5py
# import PIL.Image as Image
# from matplotlib import cm as CM


# #partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
# def gaussian_filter_density(img,points):
#     '''
#     This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

#     points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
#     img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

#     return:
#     density: the density-map we want. Same shape as input image but only has one channel.

#     example:
#     points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
#     img_shape: (768,1024) 768 is row and 1024 is column.
#     '''
#     img_shape=[img.shape[0],img.shape[1]]
#     print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
#     density = np.zeros(img_shape, dtype=np.float32)
#     gt_count = len(points)
#     if gt_count == 0:
#         return density

#     leafsize = 2048
#     # build kdtree
#     tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
#     # query kdtree
#     distances, locations = tree.query(points, k=4)

#     print ('generate density...')
#     for i, pt in enumerate(points):
#         pt2d = np.zeros(img_shape, dtype=np.float32)
#         if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
#             pt2d[int(pt[1]),int(pt[0])] = 1.
#         else:
#             continue
#         if gt_count > 1:
#             sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
#         else:
#             sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
#         density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
#     print ('done.')
#     return density


# # test code
# if __name__=="__main__":
#     # show an example to use function generate_density_map_with_fixed_kernel.
#     root = '/mnt/home/hks/平铺/DMNet-master/DMNet-master/MCNN-pytorch-master/MCNN-pytorch-master/ShanghaiTech_Crowd_Counting_Dataset'
    
#     # now generate the ShanghaiA's ground truth
#     part_A_train = os.path.join(root,'part_A_final/train_data','images')
#     part_A_test = os.path.join(root,'part_A_final/test_data','images')
#     # part_B_train = os.path.join(root,'part_B_final/train_data','images')
#     # part_B_test = os.path.join(root,'part_B_final/test_data','images')
#     path_sets = [part_A_train,part_A_test]
    
#     img_paths = []
#     for path in path_sets:
#         for img_path in glob.glob(os.path.join(path, '*.jpg')):
#             img_paths.append(img_path)
    
#     for img_path in img_paths:
#         print(img_path)
#         mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
#         img= plt.imread(img_path)#768行*1024列
#         k = np.zeros((img.shape[0],img.shape[1]))
#         points = mat["image_info"][0,0][0,0][0] #1546person*2(col,row)
#         k = gaussian_filter_density(img,points)
#         # plt.imshow(k,cmap=CM.jet)
#         # save density_map to disk
#         np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth'), k)
    
#     '''
#     #now see a sample from ShanghaiA
#     plt.imshow(Image.open(img_paths[0]))
    
#     gt_file = np.load(img_paths[0].replace('.jpg','.npy').replace('images','ground_truth'))
#     plt.imshow(gt_file,cmap=CM.jet)
    
#     print(np.sum(gt_file))# don't mind this slight variation
#     '''
import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM

# 从txt文件中读取标注信息
def read_annotations_from_txt(txt_path):
    points = []
    with open(txt_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            # 假设x和y是前两个值
            x, y = map(int, values[0:2])
            points.append([y, x])  # 交换x和y，因为图像坐标中y是行，x是列
    return points

def gaussian_filter_density(img, points):
    img_shape = [img.shape[0], img.shape[1]]
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), " gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    distances, locations = tree.query(points, k=4)

    print('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[0]) < img_shape[0] and int(pt[1]) < img_shape[1]:
            pt2d[int(pt[0]), int(pt[1])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(img.shape) / 2. / 2.  # case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

if __name__ == "__main__":
    root = '/mnt/home/hks/平铺/DMNet-master/DMNet-master/MCNN-pytorch-master/MCNN-pytorch-master/dataset'
    
    part_A_train = os.path.join(root, 'part_A/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A/test_data', 'images')
    path_sets = [part_A_train, part_A_test]
    
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    
    for img_path in img_paths:
        print(img_path)
        # 构造对应的txt文件路径
        txt_path = img_path.replace('.jpg', '.txt').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_')
        points = read_annotations_from_txt(txt_path)
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        k = gaussian_filter_density(img, points)
        np.save(img_path.replace('.jpg', '.npy').replace('images', 'dens'), k)
        # k = (k - k.min()) / (k.max() - k.min())  # 归一化到0-1范围
        # k = np.uint8(CM.jet(k) * 255)  # 应用colormap并转换为0-255范围
        
        # # 使用 PIL 保存密度图为 PNG 文件
        # density_map_image = Image.fromarray(k)
        # density_map_path_png = img_path.replace('.jpg', '_density.png').replace('images', 'den')
        # density_map_image.save(density_map_path_png)

    # 以下代码用于查看生成的密度图样本
    # plt.imshow(Image.open(img_paths[0]))
    # gt_file = np.load(img_paths[0].replace('.jpg', '.npy').replace('images', 'ground_truth'))
    # plt.imshow(gt_file, cmap=CM.jet)
    # plt.show()
    # print(np.sum(gt_file))  # don't mind this slight variation
