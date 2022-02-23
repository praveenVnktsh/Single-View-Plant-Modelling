import torch
from functions import *
from torchvision.transforms import transforms
from tqdm import tqdm
from model import Model

def loadAndProcess(im):
    height, width = im.shape[:2]
    

    if min(height, width) == height:
        height = int(width*(640/height))
        height -= height % step
        im = cv2.resize(im, (height, 640))
    else:
        width = int(height*(640/width))
        width -= width % step
        im = cv2.resize(im, (640, width))
    im = im.astype(float)/255.0
    # mask = threshold(im, lg= np.array([25, 40, 40]), ug = np.array([86, 255, 255]))
    # mask = grabcut(im, mask)

    height, width = im.shape[:2]
    # transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.ToTensor(),
    #     ])
    image = im.copy()
    im : torch.tensor = torch.tensor(im, dtype = torch.double)

    patches = im.data.unfold(0, step, step).unfold(1, step, step)
    nh, nw = patches.shape[0:2]
    patches  : torch.tensor= patches.flatten(start_dim=0, end_dim=1).double()
    # patches = torch.transpose(patches, 1, 3)
    # patches = torch.transpose(patches, 1, 2)
    print(patches.shape)
    return patches, image, (nh, nw)







def predict(patches, model):
    batchsize = 10
    for i in tqdm(range(0, patches.shape[0], batchsize)):
        p = patches[i * batchsize : (i + 1) * batchsize]
        pred = model(p)
    return pred


def postProcess(pred, image, nh, nw):
    height, width = image.shape[:2]
    finalimg = np.zeros((height, width, 3),dtype=  np.uint8)

    print(patches.numel(), finalimg.size)

    
    for i in range(nh):
        for j in range(nw):
            im :torch.tensor = patches[-(i * nw + j)]
            im = torch.transpose(im, 0, 2)
            im = torch.transpose(im, 0, 1)
            im = im.numpy()
            print(np.max(im))
            # imm = cv2.resize(patches[i].numpy(), (0, 0), fx = 4, fy = 4)
            finalimg[i*step:i*step+step, j*step:j*step+step] = im
            cv2.imshow('a', (finalimg*255).astype(np.uint8))
            if cv2.waitKey(1) == ord('q'):
                exit()

    # finalimg[finalimg > 1] = 1
    return finalimg

if __name__ == '__main__':
    


    import torch
    import os

    import glob

    import sys
  
    
    if len(sys.argv) != 3: 
        base = r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\Dataset\valdata/images/'
        model_path = r'finalModel\100_epoch=99-step=12199_1.ckpt'
        render=  True
    else:
        base = sys.argv[1]
        model_path = sys.argv[2]
        render = False
    print('!WARNING: Only PNG/JPG works!')
    print("Input Folder:", base)
    print("Model Path:", model_path)

    step = 100
    model = Model.load_from_checkpoint(model_path).double()
    model.eval()
    import os

    os.makedirs('outputs/', exist_ok = True)
    
    files = list(glob.glob(base + '*.png')) + list(glob.glob(base + '*.jpg'))
    print(files, 'Files found')
    for i, path in enumerate(files):
        print('Predicting', path)
        im = cv2.imread(path)
        patches, image, (nh, nw) = loadAndProcess(im)
        preds = predict(patches, model)
        finalImg = postProcess(patches, image, nh, nw)
        cv2.waitKey(0)
        exit()
        cv2.imwrite(f'outputs/{os.path.basename(path)}', ( finalImg * 255).astype(np.uint8))
        
        


