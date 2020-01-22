#make flying object data with single objects
import numpy as np
import random
import h5py

def make_objdict():

    utristr = '''
.....@@
....@@@
...@@@@
..@@@@@
.@@@.@@
@@@..@@
@@@..@@
.@@@.@@
..@@@@@
...@@@@
....@@@
.....@@
'''

    sqstr = '''
@@@@@@@
@@@@@@@
@@...@@
@@...@@
@@...@@
@@@@@@@
@@@@@@@
'''

    # array for square object
    sqlist = sqstr.split()
    sqarr = []
    for line in sqlist:
        row = [1.0 if c=='@' else 0.0 for c in line]
        sqarr.append(row)
    sq = np.array(sqarr)
    #print(sq)

    # array for up, down traingles
    utrilist = utristr.split()
    utriarr = []
    for line in utrilist:
        row = [1.0 if c=='@' else 0.0 for c in line]
        utriarr.append(row)
    utriarr = np.array(utriarr)
    utri = np.transpose(utriarr)
    dtri = np.flip(utri, 0)
    #print(utri)
    #print(dtri)
    
    objdict = {'sq':sq, 'utri':utri, 'dtri':dtri}
    return objdict

def stamp_object(base, obj, cy, cx):

    # Obj center location
    # cy: 0~27
    # cx: 0~27
    h,w = obj.shape
    p = 8 # pad size
    by1 = cy - h//2
    by2 = cy + (h-1)//2
    bx1 = cx - w//2
    bx2 = cx + (w-1)//2
    pad = np.pad(base, p, mode='constant')
    pad[p+by1:p+by2+1, p+bx1:p+bx2+1] += obj
    return pad[p:p+28, p:p+28]

def display_img(img):
    # display 28 x 28 x 1 image
    W, H = (28, 28)
    s = ""
    for y in range(H):
        for x in range(W):
            v = img[y, x]
            if(v < 0.5):
                c = '.'
            else:
                c = '@'
            s = s+c+" "
        s=s+"\n"
    print(s) 

def generate_single(objdict):
    # generate single random image
    obj = random.choice(objdict.values())
    cy = random.randint(0,27)
    cx = random.randint(0,27)
    base = np.zeros((28, 28))
    img = stamp_object(base, obj, cy, cx)
    return img

def generate_data(data_size):
    objdict = make_objdict()
    imgs = []
    for i in range(data_size):
        img = generate_single(objdict)
        imgs.append(np.reshape(img, (28, 28, 1)))
    data = np.array(imgs)
    return data

def main():
    data_size = 5000
    data = generate_data(data_size)
    print(data.shape)
    with h5py.File('data/single/single.h5', 'w') as f:
        f.create_dataset('features', data=data)


main()
