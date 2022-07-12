# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 读取图片，转成矩阵
# img=plt.imread('a3fingerprint.jpg')
# im=np.array(img)
#
# # 矩阵大小
# l=len(im)
# w=len(im[0])
#
# # 初始阈值
# zmin=np.min(im)
# zmax=np.max(im)
# t0=int((zmin+zmax)/2)
#
# t1=0
# res1=0
# res2=0
# s1=0
# s2=0
#
# while abs(t0-t1)>0:
#     for i in range(0,l-1):
#         for j in range(0,w-1):
#             if im[i,j]<t0:
#                 res1=res1+im[i,j]
#                 s1=s1+1
#             elif im[i,j]>t0:
#                 res2=res2+im[i,j]
#                 s2=s2+2
#     avg1=res1/s1
#     avg2=res2/s2
#     res1=0
#     res2=0
#     s1=0
#     s2=0
#     t1=t0
#     t0=int((avg1+avg2)/2)
#
# # 阈值化
# # # 小于阈值t0的像素置0，大于阈值的像素置255
# im=np.where(im[...,:]<t0,0,255)
#
# # 显示
# # # 原图
# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(img,cmap='gray')
# plt.title('original')
#
# # # #原图直方图
# # plt.figure()
# plt.subplot(2,2,2)
# plt.hist(img.ravel(),256)
# plt.title('hist')
# plt.axvline(t0)
# plt.text(25,6100,'best threshold:{}'.format(t0),size=15,alpha=0.8)
#
# # # # 阈值分割后的图
# # plt.figure()
# plt.subplot(2,2,3)
# plt.imshow(Image.fromarray(im),cmap='gray')
# plt.title('new')
#
# # # 阈值分割后的图的直方图
# # plt.figure()
# plt.subplot(2,2,4)
# plt.hist(im.ravel(),256)
# plt.title('hist')
#
# plt.show()

print('sec1')

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 读取图片，转成矩阵
img=plt.imread('a3color.jpg')
im=np.array(img[:,:,2])

# 矩阵大小
l=len(im)
w=len(im[0])

# 初始阈值
zmin=np.min(im)
zmax=np.max(im)
t0=int((zmin+zmax)/2)

t1=0
res1=0
res2=0
s1=0
s2=0

while abs(t0-t1)>0:
    for i in range(0,l-1):
        for j in range(0,w-1):
            if im[i,j]<t0:
                res1=res1+im[i,j]
                s1=s1+1
            elif im[i,j]>t0:
                res2=res2+im[i,j]
                s2=s2+2
    avg1=res1/s1
    avg2=res2/s2
    res1=0
    res2=0
    s1=0
    s2=0
    t1=t0
    t0=int((avg1+avg2)/2)

# 阈值化
# # 小于阈值t0的像素置0，大于阈值的像素置255
im=np.where(im[...,:]<t0,0,255)

# 显示
# # 原图
plt.figure()
plt.subplot(2,2,1)
plt.imshow(img,cmap='gray')
plt.title('original')

# # #原图直方图
# plt.figure()
plt.subplot(2,2,2)
plt.hist(img.ravel(),256)
plt.title('hist')
plt.axvline(t0)
plt.text(25,6100,'best threshold:{}'.format(t0),size=15,alpha=0.8)

# # # 阈值分割后的图
# plt.figure()
plt.subplot(2,2,3)
plt.imshow(Image.fromarray(im),cmap='gray')
plt.title('new')

# # 阈值分割后的图的直方图
# plt.figure()
plt.subplot(2,2,4)
plt.hist(im.ravel(),256)
plt.title('hist')

plt.show()




print('ok')