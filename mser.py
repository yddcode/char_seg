import cv2
import numpy as np
def non_max_suppression_fast(boxes, overlapThresh):
    # 空数组检测
    if len(boxes) == 0:        return [] 
        # 将类型转为float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    pick = [] 
        # 四个坐标数组
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    area = (x2 - x1 + 1) * (y2 - y1 + 1) # 计算面积数组
    idxs = np.argsort(y2) # 返回的是右下角坐标从小到大的索引值
 
        # 开始遍历删除重复的框
    while len(idxs) > 0:                # 将最右下方的框放入pick数组
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i) 
                # 找到剩下的其余框中最大的坐标x1y1，和最小的坐标x2y2,
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]]) 
                # 计算重叠面积占对应框的比例
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]] 
        # 如果占比大于阈值，则删除
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))    
        return boxes[pick].astype("int")

img = cv2.imread('mu_bseg/img/tl25_Image_00762.jpg', 1)
vis = img.copy() # 用于绘制矩形框图
orig = img.copy() # 用于绘制不重叠的矩形框图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 得到灰度图
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  # RETR_TREE
contours, hierarchy = cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)  
# <class 'list'>
# <class 'numpy.ndarray'>
# 1532
# 4
# 36
# <class 'numpy.ndarray'>
# 3
# 2
# (1, 1532, 4)
print (type(contours))
print (type(contours[0]))
print (len(contours))
print (len(contours[0]))
print (len(contours[1]))

print (type(hierarchy))  
print (hierarchy.ndim)  
print (hierarchy[0].ndim)  
print (hierarchy.shape)  
for i in range(0,len(contours)-2,3):
    # if i % 2 == 0:
    cv2.drawContours(vis,contours,i,(0,0,255),3)  
    # else:
    cv2.drawContours(vis,contours,(i+1),(0,255,0),3) 
    cv2.drawContours(vis,contours,(i+2),(255,0,0),3)  
cv2.namedWindow("vis",0)
cv2.resizeWindow("vis", 800, 800) # 限定显示图像的大小
cv2.imshow('vis', vis)

mser = cv2.MSER_create() # 得到mser算法对象
regions, _ = mser.detectRegions(gray) # 获取文本区域
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions] # 绘制文本区域
cv2.polylines(img, hulls, 1, (0, 0, 255), 5)
cv2.namedWindow("img",0)
cv2.resizeWindow("img", 800, 800) # 限定显示图像的大小
cv2.imshow('img', img)


keep = []# 绘制目前的矩形文本框
for c in hulls:
    x, y, w, h = cv2.boundingRect(c)
    keep.append([x, y, x + w, y + h])
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)            
print("[x] %d initial bounding boxes" % (len(keep)))
cv2.namedWindow("hulls",0)
cv2.resizeWindow("hulls", 800, 1000)
cv2.imshow("hulls", vis)# 筛选不重复的矩形框
keep2=np.array(keep)
pick = non_max_suppression_fast(keep2, 0.2)
print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.namedWindow("After NMS",0)
cv2.resizeWindow("After NMS", 800, 1000)
cv2.imshow("After NMS", orig)

cv2.waitKey(0)
cv2.destroyAllWindows()
