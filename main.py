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
from utils_dataset import lines

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
    batch_size=1,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
)

model = models.__dict__[args.arch](args)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint['state_dict'], strict=True)
print("Pretrained weights loaded!")
# model = model.cuda()
model.train()

viz = False

criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)

# with torch.no_grad():
for i, (pc1, pc2, generated_data, box1, box2, skip) in enumerate(val_loader):
    
    if(skip):
        continue
    
    box1 = box1.cuda()
    box2 = box2.cuda()
    
    output = model(pc1, pc2, generated_data)

    # output = output.data.cpu().numpy()
    
    output_mean_translation = torch.mean(output,axis=2)
    translated_box1 = box1 + output_mean_translation
    
    # translated_box1 = translated_box1.data.numpy()
    translated_box1 = translated_box1.view(translated_box1.size(0), -1)
    box2 = box2.view(box2.size(0), -1)
    
    loss = criterion(translated_box1, box2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if(i%5 == 0):
        print("Loss:", loss.item())
        
    
    
    if(i%100 == 0):
        
        translated_box1 = translated_box1.view(-1,8,3).data.cpu().numpy()[0]

        colors = [[0, 1, 0] for i in range(len(lines))]
        line_set_translated_box1 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(translated_box1),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set_translated_box1.colors = o3d.utility.Vector3dVector(colors)
        
        non_translated_box1 = box1.view(-1,8,3)[0] #+ output_mean_translation
        non_translated_box1 = non_translated_box1.data.cpu().numpy()
        
        colors = [[0, 0, 1] for i in range(len(lines))]
        line_set_non_translated_box1 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(non_translated_box1),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set_non_translated_box1.colors = o3d.utility.Vector3dVector(colors)
        
        non_translated_box2 = box2.view(-1,8,3)[0] #+ output_mean_translation
        non_translated_box2 = non_translated_box2.data.cpu().numpy()
        
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set_non_translated_box2 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(non_translated_box2),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set_non_translated_box2.colors = o3d.utility.Vector3dVector(colors)
        
        
        projected = pc1 + output.data.cpu().numpy()
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
        
        o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, line_set_translated_box1, line_set_non_translated_box1, line_set_non_translated_box2])


    
    
    