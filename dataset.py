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



#### Returns a list of [index1, index2, ID_in_sequence]
def populate_train_list(basedir, sequence):
    
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
                
    
    tracklets_list = []
    
    for key in data_dict.keys():
        
        for i in range(len(data_dict[key])-1):
            
            if(data_dict[key][i+1] - data_dict[key][i] == 1):   
                tracklets_list.append([data_dict[key][i],data_dict[key][i+1],key])
                
    return tracklets_list            

    
        
    
    
    

class track_and_flow_dataset(data.Dataset):
    
    def __init__(self, basedir, sequence, transform=None, gen_func=None, args=None, viz=False):
        
        self.data = pykitti.tracking(basedir, sequence)
        calib_location = basedir + "calib/" + sequence + '.txt'
        self.calibs = read_calib_file(calib_location)

        self.R0 = self.calibs['R_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        self.V2C = self.calibs['Tr_velo_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        
        self.tracklets_list = populate_train_list(basedir, sequence)
        
        self.transform = transform
        self.gen_func = gen_func
        self.num_points = args.num_points
        self.remove_ground = args.remove_ground
    
        self.viz = viz
        
    def __getitem__(self, index):
        
        idx0, idx1, object_id = self.tracklets_list[index]
        velo0 = self.data.get_velo(idx0)[:,:3]
        velo1 = self.data.get_velo(idx1)[:,:3]
        
        objects0 = self.data.get_objects(idx0)
        objects1 = self.data.get_objects(idx1)
        
        obj_to_track0 = [obj for obj in objects0 if (obj.track_id==object_id and obj.type != 'DontCare')][0]
        obj_to_track1 = [obj for obj in objects1 if (obj.track_id==object_id and obj.type != 'DontCare')][0]
        
        bounding_box_3d, R, translation = compute_box_3d(obj_to_track0)
        object0_3d_bbox_in_velo_frame = project_rect_to_velo(bounding_box_3d, self.R0, self.C2V)
        box_containment_checker = boxContainmentChecker(object0_3d_bbox_in_velo_frame)
        filtered_points0 = velo0[box_containment_checker.check_batch(velo0)]
        
        bounding_box_3d, R, translation = compute_box_3d(obj_to_track1)
        object1_3d_bbox_in_velo_frame = project_rect_to_velo(bounding_box_3d, self.R0, self.C2V)
        box_containment_checker = boxContainmentChecker(object1_3d_bbox_in_velo_frame)
        filtered_points1 = velo1[box_containment_checker.check_batch(velo1)]
        
        len0 = len(filtered_points0)
        len1 = len(filtered_points1)
        minLen = min(len0, len1)
        
        ##### NO IDEA WHY THIS WORKS ##### JUST MAKE SURE TO DO THIS
        # filtered_points0[:,:3] *= -1.
        # filtered_points1[:,:3] *= -1.
        # print(filtered_points0[:,1].shape)
        filtered_points0 = np.vstack([filtered_points0[:,1], filtered_points0[:,2], filtered_points0[:,0]]).transpose()
        filtered_points1 = np.vstack([filtered_points1[:,1], filtered_points1[:,2], filtered_points1[:,0]]).transpose()
        # print(filtered_points0.shape)
        # brak
        #### Start processing ####
        pc1_transformed, pc2_transformed, sf = self.transform([filtered_points0[:minLen], filtered_points1[:minLen]])
        # pc1_transformed, pc2_transformed, sf = self.transform([velo0[:2048], velo1[:2048]])
        
        pc1, pc2, sf, generated_data = self.gen_func([pc1_transformed, pc2_transformed, sf])

        
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
            
        return pc1, pc2, generated_data
        
    
    def __len__(self):
        return len(self.tracklets_list)
    
    
    
    

if __name__ == "__main__":
    
    basedir = "/home/mayank/Data/KITTI/training/"
    sequence = "0010"
    dataset = track_and_flow_loader(basedir, sequence)
    dataset.__getitem__(500)