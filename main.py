import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import transforms
import dataset
import models
import cmd_args
import open3d as o3d


args = cmd_args.parse_args_from_yaml("/home/mayank/Mayank/TrackThisFlow/configs/test_ours_KITTI.yaml")

basedir = "/home/mayank/Data/KITTI/training/"
sequence = "0000"

val_dataset = dataset.track_and_flow_dataset(basedir,sequence,
    transform=transforms.ProcessData(args.data_process,
                                        args.num_points,
                                        args.allow_less_points),
    gen_func=transforms.GenerateDataUnsymmetric(args),
    args=args
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
)

model = models.__dict__[args.arch](args)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint['state_dict'], strict=True)
print("Pretrained weights loaded!")
# model = model.cuda()
model.eval()

with torch.no_grad():
    for i, (pc1, pc2, generated_data) in enumerate(val_loader):
     
        output = model(pc1, pc2, generated_data)
        # print(output)
        output = output.data.cpu().numpy()
        
        projected = pc1 + output
        projected = projected.data.cpu().numpy()[0].transpose()
        pc1 = pc1.data.cpu().numpy()[0].transpose()
        pc2 = pc2.data.cpu().numpy()[0].transpose()
        
        
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1)
        pcd1.paint_uniform_color((0.0,0.0,1.0))
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2)
        pcd2.paint_uniform_color((1.0,0.0,0.0))
        
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(projected)
        pcd3.paint_uniform_color((0.0,1.0,0.0))
        
        o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
