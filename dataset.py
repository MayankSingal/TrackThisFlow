import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

import pykitti
import open3d as o3d
from utils_dataset import roty, compute_box_3d, cart2hom, read_calib_file, project_rect_to_velo, inverse_rigid_trans, convert_to_local, boxContainmentChecker, lines

sequences = ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009','0010','0011','0012','0013','0014','0015','0016','0017','0018','0019','0020']

#### Returns a list of [index1, index2, ID_in_sequence]
def populate_train_list(basedir):
    
    tracklets_list = []

    for sequence in sequences:
        
        data = pykitti.tracking(basedir, sequence)
        
        data_dict = {}
        
        for i in range(len(data)):
            objects = data.get_objects(i)
            for obj in objects:
                if(obj.type=='DontCare'): 
                    continue
                if(obj.track_id in data_dict.keys()):
                    data_dict[obj.track_id].append(i)
                else:
                    data_dict[obj.track_id] = [i]
                    
        for key in data_dict.keys():
            
            for i in range(len(data_dict[key])-1):
                
                if(data_dict[key][i+1] - data_dict[key][i] == 1):   
                    tracklets_list.append([data_dict[key][i],data_dict[key][i+1],key, sequence])
                
    return tracklets_list            

    
        
    
    
    

class track_and_flow_dataset(data.Dataset):
    
    def __init__(self, basedir, transform=None, gen_func=None, args=None, viz=False):
        
        self.data = [pykitti.tracking(basedir, sequence) for sequence in sequences]
        calib_locations = [basedir + "calib/" + sequence + '.txt' for sequence in sequences]
        self.calibs = [read_calib_file(calib_location) for calib_location in calib_locations]

        self.R0 = [calibs_['R_rect'] for calibs_ in self.calibs]
        self.R0 = [np.reshape(R0_,[3,3]) for R0_ in self.R0]

        self.V2C = [calibs_['Tr_velo_cam'] for calibs_ in self.calibs]
        self.V2C = [np.reshape(V2C_, [3,4]) for V2C_ in self.V2C]
        self.C2V = [inverse_rigid_trans(V2C_) for V2C_ in self.V2C]
        
        self.tracklets_list = populate_train_list(basedir)
        
        self.transform = transform
        self.gen_func = gen_func
        self.num_points = args.num_points
        self.remove_ground = args.remove_ground
    
        self.viz = viz
        
    def __getitem__(self, index):
        
        idx0, idx1, object_id, sequence = self.tracklets_list[index]
        sequence = sequences.index(sequence)
        
        velo0 = self.data[sequence].get_velo(idx0)[:,:3]
        velo1 = self.data[sequence].get_velo(idx1)[:,:3]
        
        objects0 = self.data[sequence].get_objects(idx0)
        objects1 = self.data[sequence].get_objects(idx1)
        
        obj_to_track0 = [obj for obj in objects0 if (obj.track_id==object_id and obj.type != 'DontCare')][0]
        obj_to_track1 = [obj for obj in objects1 if (obj.track_id==object_id and obj.type != 'DontCare')][0]
        
        bounding_box_3d, R, translation = compute_box_3d(obj_to_track0)
        object0_3d_bbox_in_velo_frame = project_rect_to_velo(bounding_box_3d, self.R0[sequence], self.C2V[sequence])
        box_containment_checker = boxContainmentChecker(object0_3d_bbox_in_velo_frame)
        filtered_points0 = velo0[box_containment_checker.check_batch(velo0)]
        
        bounding_box_3d, R, translation = compute_box_3d(obj_to_track1)
        object1_3d_bbox_in_velo_frame = project_rect_to_velo(bounding_box_3d, self.R0[sequence], self.C2V[sequence])
        box_containment_checker = boxContainmentChecker(object1_3d_bbox_in_velo_frame)
        filtered_points1 = velo1[box_containment_checker.check_batch(velo1)]
        
        len0 = len(filtered_points0)
        len1 = len(filtered_points1)
        minLen = min(min(len0, len1),1024)
        
        random.shuffle(filtered_points0)
        random.shuffle(filtered_points1)
        
        ##### NO IDEA WHY THIS WORKS ##### JUST MAKE SURE TO DO THIS

        filtered_points0 = np.vstack([filtered_points0[:,1], filtered_points0[:,2], filtered_points0[:,0]]).transpose()
        filtered_points1 = np.vstack([filtered_points1[:,1], filtered_points1[:,2], filtered_points1[:,0]]).transpose()

        # print("C",len(filtered_points0), len(filtered_points1))

        #### Start processing ####
        pc1_transformed, pc2_transformed, sf = self.transform([filtered_points0[:minLen], filtered_points1[:minLen]])
        # pc1_transformed, pc2_transformed, sf = self.transform([velo0[:2048], velo1[:2048]])
        
        # print("B",len(pc1_transformed), len(pc2_transformed))

        pc1, pc2, sf, generated_data = self.gen_func([pc1_transformed, pc2_transformed, sf])
        
        #### Change axes for bboxes also
        object0_3d_bbox_in_velo_frame = np.vstack([object0_3d_bbox_in_velo_frame[:,1], object0_3d_bbox_in_velo_frame[:,2], object0_3d_bbox_in_velo_frame[:,0]]).transpose()
        object1_3d_bbox_in_velo_frame = np.vstack([object1_3d_bbox_in_velo_frame[:,1], object1_3d_bbox_in_velo_frame[:,2], object1_3d_bbox_in_velo_frame[:,0]]).transpose()

        
        if(self.viz == True):
            
            pcd0 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(filtered_points0)
            pcd0.paint_uniform_color((0.0,1.0,0))
            colors = [[0, 1, 0] for i in range(len(lines))]
            line_set0 = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(object0_3d_bbox_in_velo_frame),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set0.colors = o3d.utility.Vector3dVector(colors)
            
            
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(filtered_points1)
            pcd1.paint_uniform_color((1.0,0.0,1.0))
            colors = [[1, 0, 1] for i in range(len(lines))]
            line_set1 = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(object1_3d_bbox_in_velo_frame),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set1.colors = o3d.utility.Vector3dVector(colors)
            

            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(velo0)
            pcd2.paint_uniform_color((1.0,0.0,0.0))
            
            o3d.visualization.draw_geometries([pcd0, pcd1, pcd2, line_set0, line_set1])
        
        # print("A",len(pc1), len(pc2))
        skip = 0
        if(pc1 is None or pc2 is None or generated_data is None or object0_3d_bbox_in_velo_frame is None or object1_3d_bbox_in_velo_frame is None or minLen < 128):
            pc1 = 1
            pc2 = 1
            generated_data = 1
            object0_3d_bbox_in_velo_frame = 1
            object1_3d_bbox_in_velo_frame = 1
            skip = 1
            
        return pc1, pc2, generated_data, object0_3d_bbox_in_velo_frame, object1_3d_bbox_in_velo_frame, skip
        
    
    def __len__(self):
        return len(self.tracklets_list)
    
    
    
    
    

if __name__ == "__main__":
    
    basedir = "/home/mayank/Data/KITTI/training/"
    sequence = "0010"
    dataset = track_and_flow_loader(basedir, sequence)
    dataset.__getitem__(500)