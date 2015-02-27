import cv2
import matplotlib.pyplot as plt

class TestClass():
    def __init__(self):
        self.point = []
    
    def __onclick__(self,click):
        self.point.append((click.xdata,click.ydata))
        return self.point
        
    def getCoord(self,img):
        fig = plt.figure()
        plt.imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.point
    
    def getBox(self,img):
        self.point = []
        vertices = self.getCoord(img)[-2:]
        x0 = int(min(vertices[0][0],vertices[1][0]))
        y0 = int(min(vertices[0][1],vertices[1][1]))
        xf = int(max(vertices[0][0],vertices[1][0]))
        yf = int(max(vertices[0][1],vertices[1][1]))
        roi = {'x0':x0,'y0':y0,'xf':xf,'yf':yf,'roi':img[y0:yf,x0:xf]}
        return roi
    
    def setDrop(self,img):
        #check: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
        #select the bounding box
        #select the outermost points of the edges (left/right)
        #for each each, select its bounding box template
        pass
