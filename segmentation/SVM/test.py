import torch
from functions import *

def predict(image):

    mask = threshold(image, lg= np.array([25, 40, 40]), ug = np.array([86, 255, 255]))
    

    mask = grabcut(image, mask)
    cv2.imshow('a', mask)
    cv2.waitKey(1)

    hWinSize = 16

    mask = cv2.copyMakeBorder(mask, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
    image = cv2.copyMakeBorder(image, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)

    x, y, indices = sampleGrid(mask, step = 6, viz = False)
    leaves = np.zeros_like(mask)
    stems = np.zeros_like(mask)
    ogmask = mask.copy()
    mask = np.stack((mask, mask, mask), axis = 2)
    plantSeg = ((mask.astype(float)/255) * image).astype(np.uint8)

    
    

    classWindow = 5
    for point in list(zip(y, x)):
        yy, xx = point
        window = image[yy - hWinSize : yy + hWinSize, xx - hWinSize : xx + hWinSize]
        H = hog(window)
        label = model.predict(H.reshape(1, 324))
        if label == 1:
            stems[yy - classWindow : yy + classWindow, xx - classWindow : xx + classWindow] = 255
            leaves[yy - classWindow : yy + classWindow, xx - classWindow : xx + classWindow] = 0
        else:
            stems[yy - classWindow : yy + classWindow, xx - classWindow : xx + classWindow] = 0
            leaves[yy - classWindow : yy + classWindow, xx - classWindow : xx + classWindow] = 255

    
    stems = stems & ogmask
    leaves = leaves & ogmask
    stems = stems[hWinSize : -hWinSize, hWinSize : -hWinSize]
    leaves = leaves[hWinSize : -hWinSize, hWinSize : -hWinSize]
    print('Complete')
    
    return stems, leaves


if __name__ == '__main__':
    base = r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\Dataset\valdata/images/'
    import torch
    import os

    import glob
    import os
    # modelname = f'RBF SVM_91_20000'
    modelname = 'Nearest Neighbors_90_30000'
    model = torch.load(f'binary/{modelname}.pt')
    os.makedirs(f'outputs/{modelname}/', exist_ok= True)

    for path in glob.glob(base + '*.png'):
        img = cv2.imread(path)
        # img = cv2.resize(img, (0, 0), fx = 0.2, fy = 0.2)
        
        stems, leaves = predict(img)
        # img[:, :, 0][leaves == 255] = 255
        # img[:, :, 2][stems == 255] = 255
        # img[:, :, 0][stems == 0] = 0
        # img[:, :, 1][stems == 0] = 0
        # img[:, :, 2][stems == 0] = 0
        # img[img == 0] =255
        # cv2.imshow('a', img)
        cv2.imwrite(f'outputs/{modelname}/{os.path.basename(path)}', stems)
        if cv2.waitKey(1) == ord('q'):
            exit()


