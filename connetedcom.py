import cv2
import numpy as np
 

img = cv2.imread("char_seg/img.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


ret, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,4))
bin_clo = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations = 1)
 
cv2.namedWindow("img",0)
cv2.resizeWindow("img", 700, 1000) # 限定显示图像的大小
cv2.imshow('img', bin_clo)

'''水平投影'''
def getHProjection(image):
    hProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w)=image.shape
    #长度与图像高度一致的数组
    h_ = [0]*h
    #循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 255:
                h_[y]+=1
    #绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y,x] = 255
    cv2.namedWindow("hProjection2",0)
    cv2.resizeWindow("hProjection2", 700, 1000) # 限定显示图像的大小
    cv2.imshow('hProjection2',hProjection)
    return h_
  
#水平投影
H = getHProjection(bin_clo)
(h,w) = binary.shape
print(H)
I = []
# for i in range(len(H)):
#     if H[i]< w/2:
#       print(i)
#       I.append(i)
A = [0, 215, 430, 645, 860, 1075, 1290, 1505, 1720, 1935, 2150, 2365, 2580, 2795, 3010, 3225, 3440, 3655, 3870, 4085, 4300, 4515] 
for i in range(len(A)):
    cv2.line(img,(0,A[i]),(300,A[i]),(0,0,255),10,cv2.LINE_AA)

cv2.namedWindow("line",0)
cv2.resizeWindow("line", 700, 1000) # 限定显示图像的大小
cv2.imshow('line', img)

def getVProjection(image):
    vProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w) = image.shape
    #长度与图像宽度一致的数组
    w_ = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 255:
                w_[x]+=1
    #绘制垂直平投影图像
    for x in range(w):
        for y in range(h-w_[x],h):
            vProjection[y,x] = 255
    cv2.imshow('vProjection',vProjection)
    return w_

# Position = []
# start = 0
# H_Start = []
# H_End = []
# #根据水平投影获取垂直分割位置
# for i in range(len(H)):
#     if H[i] > 0 and start ==0:
#         H_Start.append(i)
#         start = 1
#     if H[i] <= 0 and start == 1:
#         H_End.append(i)
#         start = 0
# #分割行，分割之后再进行列分割并保存分割位置
# for i in range(len(H_Start)):
#     #获取行图像
#     cropImg = img[H_Start[i]:H_End[i], 0:w]
#     #cv2.imshow('cropImg',cropImg)
#     #对行图像进行垂直投影
#     W = getVProjection(cropImg)
#     Wstart = 0
#     Wend = 0
#     W_Start = 0
#     W_End = 0
#     for j in range(len(W)):
#         if W[j] > 0 and Wstart ==0:
#             W_Start =j
#             Wstart = 1
#             Wend=0
#         if W[j] <= 0 and Wstart == 1:
#             W_End =j
#             Wstart = 0
#             Wend=1
#         if Wend == 1:
#             Position.append([W_Start,H_Start[i],W_End,H_End[i]])
#             Wend =0
# #根据确定的位置分割字符
# for m in range(len(Position)):
#     cv2.rectangle(img, (Position[m][0],Position[m][1]), (Position[m][2],Position[m][3]), (0 ,229 ,238), 1)
# cv2.imshow('image', img)
# cv2.waitKey(0)

# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo,connectivity = 8)
 
# """
# #查看各个返回值
# print('num_labels = ',num_labels)
# print('stats = ',stats)
# print('centroids = ',centroids)
# print('labels = ',labels)
# """
 
# label_area = stats[:,-1]
# max_index = np.argmax(label_area)
 
# #label the backgroud and foreground
# height = labels.shape[0]
# width = labels.shape[1]
# for row in range(height):
#     for col in range(width):
#         if labels[row,col] == max_index:
#             gray[row,col] = 0
#         else:
#             gray[row,col] = 255
 
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
# conne = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations = 2)
 
# cv2.namedWindow('results',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('results',conne)
cv2.waitKey(0)
cv2.destroyAllWindows()
