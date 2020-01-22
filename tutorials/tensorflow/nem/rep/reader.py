import numpy as np
import h5py

def main():
    f = h5py.File('data/shapes.h5', 'r')
    train_feature = f['training']['features']
    print(train_feature.shape)
    imgs = train_feature[0, :]
    N = imgs.shape[0]
    for i in range(N):
        display_img(imgs[i])
        _ = raw_input()


def display_img(img):
    # display 28 x 28 x 1 image
    W, H = (28, 28)
    s = ""
    for y in range(H):
        for x in range(W):
            v = img[x, y, 0]
            if(v < 0.5):
                c = '.'
            else:
                c = '@'
            s = s+c+" "
        s=s+"\n"
    print(s)
                        

main()
