import glob

imgsL = sorted(glob.glob('img/L*.png'))
imgsR = sorted(glob.glob('img/R*.png'))

flag = True

for imgL, imgR in zip(imgsL, imgsR):
    print imgL[5:-4], imgR[5:-4]
    if imgL[5:-4] != imgR[5:-4]:
        flag = False

print flag
