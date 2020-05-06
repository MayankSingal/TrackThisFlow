import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable




lines = [
    
        [0,1],
        [1,2],
        [2,3],
        [3,0],
        # [3,7],
        # [0,4],
        # [4,5],
        # [4,7],
        # [5,1],
        # [5,6],
        # [6,7],
        # [2,6]
    
]

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])



def convert_to_local(points_3d, R, translation):
    points_3d = copy.copy(np.transpose(points_3d))
    
    points_3d[0,:] = points_3d[0,:] - translation.x
    points_3d[1,:] = points_3d[1,:] - translation.y
    points_3d[2,:] = points_3d[2,:] - translation.z
    
    points_3d = np.dot(np.linalg.inv(R), points_3d)
    
    return np.transpose(points_3d)
    
    
def filtered_points(obj,points_3d):

    l = obj.dimensions.length
    w = obj.dimensions.width
    h = obj.dimensions.height
    
    output_points = []
    
    print(points_3d, l, w, h)
    
    for i in range(len(points_3d)):
        if(True):
            if(points_3d[i][1] > -h and points_3d[i][1] < 0):
                if(points_3d[i][2] > -w/2 and points_3d[i][2] < w/2):
                    output_points.append(points_3d[i])
                    
    return output_points


class boxContainmentChecker():
    
    def __init__(self, box):
        
        self.p1 = box[3]
        self.p2 = box[0]
        self.p4 = box[2]
        self.p5 = box[7]
        
        self.u = self.p1-self.p2#np.cross((self.p1 - self.p4), (self.p1 - self.p5))
        self.v = self.p1-self.p4#np.cross((self.p1 - self.p2), (self.p1 - self.p5))
        self.w = self.p1-self.p5#np.cross((self.p1 - self.p2), (self.p1 - self.p4))
        
        self.u_low = np.dot(self.u, self.p1)
        self.u_high = np.dot(self.u, self.p2)
        
        self.v_low = np.dot(self.v, self.p1)
        self.v_high = np.dot(self.v, self.p4)
        
        self.w_low = np.dot(self.w, self.p1)
        self.w_high = np.dot(self.w, self.p5)
        
        # print(self.u_low, self.u_high, self.v_low, self.v_high, self.w_low, self.w_high)

        
        
    def check(self, point):
        
        if(np.dot(self.u,point) < self.u_low and np.dot(self.u,point) > self.u_high):
            # print("YO")
            if(np.dot(self.v,point) < self.v_low and np.dot(self.v,point) > self.v_high):
                # print("L")
                if(np.dot(self.w,point) < self.w_low and np.dot(self.w,point) > self.w_high):
                    # print("0")
                    return True
                
        return False
    
    def check_batch(self, points):
        
        u_projections = np.dot(points, self.u)
        v_projections = np.dot(points, self.v)
        w_projections = np.dot(points, self.w)
        
        u_filtered_positions = np.logical_and(u_projections < self.u_low, u_projections > self.u_high)
        v_filtered_positions = np.logical_and(v_projections < self.v_low, v_projections > self.v_high)
        w_filtered_positions = np.logical_and(w_projections < self.w_low, w_projections > self.w_high)
        filtered_positions = np.logical_and(np.logical_and(u_filtered_positions, v_filtered_positions), w_filtered_positions)
                
        return filtered_positions
            
        
        
        
        
        
        
        
            


def compute_box_3d(obj):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.rotation_y)    
    # R = np.eye(3)

    # 3d bounding box dimensions
    l = obj.dimensions.length
    w = obj.dimensions.width
    h = obj.dimensions.height
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))

    corners_3d[0,:] = corners_3d[0,:] + obj.location.x
    corners_3d[1,:] = corners_3d[1,:] + obj.location.y
    corners_3d[2,:] = corners_3d[2,:] + obj.location.z
    
    
    # corners_3d[0,:] = corners_3d[0,:] - obj.location.x
    # corners_3d[1,:] = corners_3d[1,:] - obj.location.y
    # corners_3d[2,:] = corners_3d[2,:] - obj.location.z
    
    
    # corners_3d = np.dot(np.linalg.inv(R), corners_3d)
    
    
    
    # # only draw 3d bounding box for objs in front of the camera
    # if np.any(corners_3d[2,:]<0.1):
    #     corners_2d = None
    #     return corners_2d, np.transpose(corners_3d)
    
    # project the 3d bounding box into the image plane
    # corners_2d = project_to_image(np.transpose(corners_3d), P);
    #print 'corners_2d: ', corners_2d
    # return corners_2d, np.transpose(corners_3d)
    return np.transpose(corners_3d), R, obj.location


def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
    return pts_3d_hom


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(' ', 1) 
            key = key.rstrip(':') 
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def project_rect_to_velo(pts_3d_rect, R0, C2V):
    ''' Input: nx3 points in rect camera coord.
        Output: nx3 points in velodyne coord.
    ''' 
    pts_3d_ref = project_rect_to_ref(pts_3d_rect, R0)
    return project_ref_to_velo(pts_3d_ref, C2V)

def project_rect_to_ref(pts_3d_rect, R0):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))

def project_ref_to_velo(pts_3d_ref, C2V):
    pts_3d_ref = cart2hom(pts_3d_ref) # nx4
    return np.dot(pts_3d_ref, np.transpose(C2V))

def project_velo_to_rect(pts_3d_velo, R0, V2C):
    pts_3d_ref = project_velo_to_ref(pts_3d_velo, V2C)
    return project_ref_to_rect(pts_3d_ref, R0)

def project_velo_to_ref(pts_3d_velo, V2C):
    pts_3d_velo = cart2hom(pts_3d_velo) # nx4
    return np.dot(pts_3d_velo, np.transpose(V2C))

def project_ref_to_rect(pts_3d_ref, R0):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))



def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

# import shutil
# import os, sys
# def save_checkpoint(state, is_best, ckpt_dir, filename='checkpoint.pth.tar'):
#     torch.save(state, os.path.join(ckpt_dir, filename))
#     if state['epoch'] % 1 == 0:
#         shutil.copyfile(
#             os.path.join(ckpt_dir, filename),
#             os.path.join(ckpt_dir, 'checkpoint_'+str(state['epoch'])+'.pth.tar'))

#     if is_best:
#         shutil.copyfile(
#             os.path.join(ckpt_dir, filename),
#             os.path.join(ckpt_dir, 'model_best.pth.tar'))

# def filter_points_inside_cube():
   