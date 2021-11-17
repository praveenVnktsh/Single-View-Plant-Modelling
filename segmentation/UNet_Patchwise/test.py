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
    return finalmask, mask[hWinSize:-hWinSize, hWinSize:-hWinSize]


if __name__ == '__main__':
    base = r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\testData/'
    import torch
    import os

    import glob

    model = Model.load_from_checkpoint(r'finalModel\epoch=78-step=59723.ckpt')
    model.eval()
    for i, path in enumerate(glob.glob(base + '*.jpg')):
        
        im = cv2.imread(path)
        if i >= 2:
            im = cv2.resize(im, (0, 0), fx = 0.15, fy = 0.15)
            thresh = 0.5
            continue
        else:
            im = cv2.resize(im, (0, 0), fx = 0.2, fy = 0.2)
            thresh = 0.8

        masks, overallmask = predict(im, model)

        cv2.imwrite(f'toModel/{i}_img.jpg', im)
        cv2.imwrite(f'toModel/{i}_stem.jpg',( masks * 255).astype(np.uint8))
        cv2.imwrite(f'toModel/{i}_mask.jpg', overallmask)
        
        temp = im.copy()
        temp[:, :, 2][masks > thresh] = 255
        cv2.imwrite(f'outputs/{i}_illu.jpg', temp)

        im[:, :, 0][masks < thresh] = 255
        im[:, :, 1][masks < thresh] = 255
        im[:, :, 2][masks < thresh] = 255
        cv2.imwrite(f'outputs/{i}.jpg', im)

        
        if cv2.waitKey(1) == ord('q'):
            exit()


