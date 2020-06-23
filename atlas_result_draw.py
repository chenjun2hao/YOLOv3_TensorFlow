import cv2
from utils.plot_utils import get_color_table, plot_one_box

'''
#0, bbox(   9,   73,   28,   92) confidence: 0.885981 classId is 1 
 #1, bbox(  61,   67,   79,   84) confidence: 0.809121 classId is 1 
 #2, bbox( 210,   68,  229,   89) confidence: 0.770393 classId is 1 
 #3, bbox( 306,   60,  334,   92) confidence: 0.706168 classId is 1 
 #4, bbox( 205,   55,  215,   66) confidence: 0.659027 classId is 1 
 #5, bbox( 142,   57,  152,   68) confidence: 0.654435 classId is 1 
 #6, bbox( 280,   53,  297,   72) confidence: 0.637229 classId is 1 
 #7, bbox( 162,   61,  187,   90) confidence: 0.635332 classId is 1 
 #8, bbox( 382,   58,  396,   73) confidence: 0.617725 classId is 1 
 #9, bbox( 108,   68,  129,   88) confidence: 0.578017 classId is 1 
 #10, bbox( 153,   57,  160,   65) confidence: 0.552839 classId is 1 
 #11, bbox( 301,   54,  319,   75) confidence: 0.502713 classId is 1 
 
  #0, bbox( 239,   61,  366,  187) confidence: 0.931187 classId is 0 
 #1, bbox( 358,   28,  477,  166) confidence: 0.931488 classId is 1 
 #2, bbox(  61,  109,  156,  239) confidence: 0.893770 classId is 1 

'''
dict_label = {0:'hat', 1:'person'}
bbox = [[239,   61,  366,  187, 0], [358,   28,  477,  166, 1], [61,  109,  156,  239, 1]]
img_path = './data/demo_data/000009.jpg'
img = cv2.imread(img_path)

for box in bbox:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    label = dict_label[box[4]]
    cv2.putText(img, label, (box[0], box[1] - 2), 0, float(2) / 3, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

cv2.imshow('Atlas500', img)
cv2.waitKey()