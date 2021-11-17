import numpy as np
import cv2
import cv2


def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def save_stereo_coefficients(path, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q):
    """ Save the stereo coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K1", K1)
    cv_file.write("D1", D1)
    cv_file.write("K2", K2)
    cv_file.write("D2", D2)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.write("E", E)
    cv_file.write("F", F)
    cv_file.write("R1", R1)
    cv_file.write("R2", R2)
    cv_file.write("P1", P1)
    cv_file.write("P2", P2)
    cv_file.write("Q", Q)
    cv_file.release()


def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def load_stereo_coefficients(path):
    """ Loads stereo matrix coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]



def scaleAndShow(im, name = 'outdoor', height = None, waitkey = 1):
    def callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y, im[y, x])
    
    cv2.namedWindow(name)
    cv2.setMouseCallback(name,callback)
    if height is not None:
        width = int(im.shape[1]*height/im.shape[0])
        im = cv2.resize(im, (width, height), interpolation= cv2.INTER_NEAREST)
    cv2.imshow(name, im)
    if cv2.waitKey(waitkey) == ord('q'):
        exit()



def stackAndShow(dic : dict, height = 600, save = '', winname = 'asdf'):
    # imshape = dic['mask'].shape
    # cv2.namedWindow(winname, cv2.WINDOW_GUI_NORMAL)
    length = len(dic)

    nStack = round(np.sqrt(length))

    row = []
    finalImg = None    
    for i, (k, v) in enumerate(dic.items()):
        if i == 0:
            shape = v.shape
        if i % nStack == 0 and i != 0:
            if finalImg  is not None:
                rowstack = np.vstack(row)
                finalImg = np.hstack((finalImg, rowstack))
                
            else:
                finalImg =np.vstack(row)
            row = []

        v = v.astype(float)
        v = (v*255/v.max()).astype(np.uint8)
        v = cv2.resize(v, (shape[1], shape[0]),cv2.INTER_NEAREST)
        if len(v.shape) == 2:
            v = np.stack((v, v, v), axis = 2)
        cv2.rectangle(v, (0, 0), (5 + len(k) * 8, 30), (50, 50, 50), -1)
        cv2.putText(v, k, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        row.append(v)



    rowstack = np.vstack(row)    

    while rowstack.shape[0] != finalImg.shape[0]:
        row.append(np.zeros_like(row[0]))        
        rowstack = np.vstack(row)    

    finalImg = np.hstack((finalImg, rowstack))

    if save != '' and save is not None:
        import os
        # os.makedirs(save.spl, exist_ok = True)
        cv2.imwrite(save, finalImg)
    scaleAndShow(finalImg, winname, height = height)

        


def scaleAndShow(im, name = 'outdoor', height = None, waitkey = 1, save = False):
    def callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y, im[y, x])
    
    cv2.namedWindow(name)
    cv2.setMouseCallback(name,callback)
    if save:
        cv2.imwrite(name + '.jpg', im)
    if height is not None:
        width = int(im.shape[1]*height/im.shape[0])
        im = cv2.resize(im, (width, height), interpolation= cv2.INTER_NEAREST)
    cv2.imshow(name, im)
    
    if cv2.waitKey(waitkey) == ord('q'):
        exit()