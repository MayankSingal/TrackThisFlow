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
from torch.utils.tensorboard import SummaryWriter


args = cmd_args.parse_args_from_yaml("/home/mayank/Mayank/TrackThisFlow/configs/test_ours_KITTI.yaml")

basedir = "/home/mayank/Data/KITTI/training/"

writer = SummaryWriter()

val_dataset = dataset.track_and_flow_dataset(basedir,
    transform=transforms.ProcessData(args.data_process,
                                        args.num_points,
                                        args.allow_less_points),
    gen_func=transforms.GenerateDataUnsymmetric(args),
    args=args
)

print("Length of dataset:", len(val_dataset))

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
)

model_checker = models.__dict__[args.arch](args)
model_checker = torch.nn.DataParallel(model_checker).cuda()
checkpoint = torch.load(args.resume)
model_checker.load_state_dict(checkpoint['state_dict'], strict=True)
print("Pretrained weights loaded!")
# model = model.cuda()
model_checker.eval() 


model = models.__dict__[args.arch](args)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint['state_dict'], strict=True)
print("Pretrained weights loaded!")
# model = model.cuda()
model.train()

viz = True

def nearest_neighbour(x, y):
    
    # r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)  # (B,N,1)
    # r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)  # (B,M,1)
    # mul = torch.matmul(xyz2, xyz1.permute(0,2,1))         # (B,M,N)
    # dist = r_xyz2 - 2 * mul + r_xyz1.permute(0,2,1)       # (B,M,N)
    x = x[0]
    y = y[0]
    
    x = x.transpose(0,1)
    y = y.transpose(0,1)
    
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.pow(x - y, 2).sum(2) 
   
    nn_dists = torch.min(dist, axis=0).values
    
    return torch.sum(nn_dists)/n

criterion = torch.nn.MSELoss().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.000005, weight_decay=0.0001)


for epoch in range(1):
    # with torch.no_grad():
    skipped = 0
    for i, (pc1, pc2, generated_data, box1, box2, skip) in enumerate(val_loader):
        
        
        if(i%1000 == 0):
            state = {
                'epoch': epoch + 1,  # next start epoch
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'min_loss': 0,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, str(i) + 'newModel.pth.tar')
            print("Model saved at iteration:", i)
        
        
        
        if(skip==1):
            skipped += 1
            continue
        
        box1 = box1.cuda()
        box2 = box2.cuda()
        
        output = model(pc1, pc2, generated_data)
        # output_checker = model_checker(pc1, pc2, generated_data)

        # output = output.data.cpu().numpy()
        pc1 = pc1.cuda()
        pc2 = pc2.cuda()
        output_mean_translation = torch.mean(output,axis=2)
        translated_box1 = box1 + output_mean_translation
        
        # output_mean_translation_checker = torch.mean(output_checker,axis=2)
        # translated_box1_checker = box1 + output_mean_translation_checker
        # translated_box1_checker = translated_box1_checker.view(translated_box1_checker.size(0),-1)
        
        # translated_box1 = translated_box1.data.numpy()
        translated_box1 = translated_box1.view(translated_box1.size(0), -1)
        box2 = box2.view(box2.size(0), -1)
        
        loss_translation = criterion(translated_box1, box2)
        loss_nn = nearest_neighbour(pc1+output, pc2)
        loss = loss_translation + loss_nn
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(i%5 == 0):
            print("Epoch", epoch, "| Iteration:", i,  " | Translation Loss:", loss_translation.item(), "| NN Loss:", loss_nn.item(), "| Total Loss:", loss.item(), " | Skipped:", skipped)
            writer.add_scalar("translation_loss", loss_translation.item(), epoch*(len(val_dataset) + i))
            writer.add_scalar("nn_loss", loss_nn.item(), epoch*(len(val_dataset)) + i)
            writer.add_scalar("total_loss", loss.item(), epoch*(len(val_dataset)) + i)            
        
        
        if(False):
            output_checker = model_checker(pc1, pc2, generated_data)
            output_mean_translation_checker = torch.mean(output_checker,axis=2)
            translated_box1_checker = box1 + output_mean_translation_checker
            translated_box1_checker = translated_box1_checker.view(translated_box1_checker.size(0),-1)
                
            translated_box1 = translated_box1.view(-1,8,3).data.cpu().numpy()[0]
            colors = [[0, 1, 0] for i in range(len(lines))]
            line_set_translated_box1 = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(translated_box1),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set_translated_box1.colors = o3d.utility.Vector3dVector(colors)
            
            translated_box1_checker = translated_box1_checker.view(-1,8,3).data.cpu().numpy()[0]
            colors = [[0, 0, 0] for i in range(len(lines))]
            line_set_translated_box1_checker = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(translated_box1_checker),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set_translated_box1_checker.colors = o3d.utility.Vector3dVector(colors)
            
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
            
            projected = pc1.data.cpu().numpy() + output.data.cpu().numpy()
            projected = projected[0].transpose()

            projected_checker = pc1.data.cpu().numpy() + output_checker.data.cpu().numpy()
            projected_checker = projected_checker[0].transpose()
            
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
            
            pcd4 = o3d.geometry.PointCloud()
            pcd4.points = o3d.utility.Vector3dVector(projected_checker)
            pcd4.paint_uniform_color((0.0,0.0,0.0))
            
            o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, pcd4, line_set_translated_box1, line_set_non_translated_box1, line_set_non_translated_box2, line_set_translated_box1_checker])


        
    
    