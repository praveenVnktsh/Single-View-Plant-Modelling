import torch
from functions import *

def predict(image):

    mask = threshold(image, lg= np.array([25, 20, 20]), ug = np.array([86, 255, 255]))
    mask = grabcut(image, mask)

    hWinSize = 16

    mask = cv2.copyMakeBorder(mask, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
    image = cv2.copyMakeBorder(image, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)

    x, y, indices = sampleGrid(mask, step = 3, viz = False)

    mask = np.stack((mask, mask, mask), axis = 2)
    plantSeg = ((mask.astype(float)/255) * image).astype(np.uint8)

    leaves = np.zeros_like(plantSeg)
    stems = np.zeros_like(plantSeg)
    

    classWindow = 10
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


    stems = stems & mask
    leaves = leaves & mask
    return stems, leaves


if __name__ == '__main__':
    base = r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData/'
    import torch
    data = torch.load(base + 'dict_tripartite.pt')
    model = torch.load('30k_085.pt')

    for i in range(data['length']):
        stems, leaves = predict(data['images'][i])
        cv2.imshow('a', stems)
        cv2.imshow('b', leaves)
        if cv2.waitKey(0) == ord('q'):
            exit()


