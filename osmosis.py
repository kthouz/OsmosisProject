import cv2
import matplotlib.pyplot as plt

class Osmosis():
    def __init__(self):
        self.point = []
    
    def __onclick__(self,click):
        self.point.append((click.xdata,click.ydata))
        return self.point
        
    def getCoord(self,img):
        """Method to get coordinates of mouse clicks"""
        fig = plt.figure()
        plt.imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.point
    
    def getBox(self,img):
        """Method to get bounding box of the region of interest.
        Input: image
        Output: dictionary of x0,y0,xf,yf and the region of interest
        """
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
        #1. select the bounding box
        #2. select the outermost points of the edges (left/right)
        #3. for each edge, select its bounding box template
        #4. use cv2.matchTemplate to find the locatioin of the edges in the frame of reference
        #5. use results from template matching to calculate the offset with respect to the origin of the initial frame
        #Output: offsets
        pass
    
    def analyzeData(self,allimgs):
        """This method uses information gathered from setting the drop, to find edges in subsequent frames"""
        #1. Get vertices location
        #2. find drop length, volume and concentration
        #3. generate figures
        #5. dump a json backup file
        #6. Try to apply pattern learning over time
