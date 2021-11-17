import torch
from functions import *
from torchvision.transforms import transforms
from tqdm import tqdm
from model import Model
def predict(image, model):

    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((30, 30)),
            transforms.ToTensor(),
        ])


    winsize = 30
    h, w, c = image.shape
    mask = threshold(image, lg= np.array([25, 40, 40]), ug = np.array([86, 255, 255]))
    

    mask = grabcut(image, mask)
    cv2.imshow('a', mask)
    cv2.waitKey(1)

    hWinSize = 16

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.copyMakeBorder(mask, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
    image = cv2.copyMakeBorder(image, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)

    x, y, indices = sampleGrid(mask, step = 10, viz = False)
    finalmask = np.zeros_like(mask).astype(np.float)
    mask = np.stack((mask, mask, mask), axis = 2)

    
    

    for point in tqdm(list(zip(y, x))):
        yy, xx = point
        window = image[yy - hWinSize : yy + hWinSize, xx - hWinSize : xx + hWinSize]
        # cv2.imshow('a', window)
        # cv2.waitKey(1)
        window = transform(window).unsqueeze(0)
        pred = model(window)
        finalmask[yy - hWinSize : yy + hWinSize, xx - hWinSize : xx + hWinSize] += cv2.resize((pred.squeeze().cpu().detach().numpy()), (32, 32))

    finalmask = finalmask[hWinSize:-hWinSize, hWinSize:-hWinSize]
    finalmask[finalmask > 1] = 1
    
    return finalmask


if __name__ == '__main__':
    base = r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\testData/'
    import torch
    import os

    import glob
    model = Model.load_from_checkpoint(r'lightning_logs\version_7\checkpoints\epoch=76-step=58211.ckpt')
    model.eval()
    for i, path in enumerate(glob.glob(base + '*.jpg')):
        # if i > 2:
        #     continue
        im = cv2.imread(path)
        im = cv2.resize(im, (0, 0), fx = 0.15, fy = 0.15)
        masks = predict(im, model)
        im[:, :, 2][masks > 0.5] = 255
        # im[:, :, 1][masks < 0.5] = 0
        # im[:, :, 0][masks < 0.5] = 0
        # im[im == 0] = 255
        cv2.imshow('a', im)
        cv2.imshow('b', masks)
        cv2.imwrite(f'outputs/{i}_1.jpg', im)
        print('Complete')
        if cv2.waitKey(1) == ord('q'):
            exit()


