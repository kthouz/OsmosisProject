import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Osmosis():
    def __init__(self,src_path, pxl2um, frame2sec, frameIncr = 1, frameRateConversion = 1):
        self.src = src_path
        self.pxl2um = pxl2um
        self.frame2sec = frame2sec
        self.frameIncr = frameIncr
        self.frameRConv = frameRateConversion
        self.imgs = self.loadImages()[0]
        self.allimgs = self.loadImages()[1]
        self.point = []
        
    def loadImages(self):
        """This methods load images to be analyzed using frameIncr"""
        try:
            allimgs = os.listdir(self.src)
            indices = np.arange(0,len(allimgs),self.frameIncr)
            imgs = [os.path.join(self.src,allimgs[i]) for i in indices]
        except:
            print "Directory not found, empty or frameIncr and number of images in directory are incompatible"
        return [imgs,allimgs]
    
    def __onclick__(self,click):
        self.point.append((int(click.xdata),int(click.ydata)))
        return self.point
        
    def getCoord(self,img,s):
        fig = plt.figure()
        plt.imshow(img)
        plt.title(s)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        return self.point
    
    def getBox(self,img,s):
        self.point = []
        vertices = self.getCoord(img,s)[-2:]
        x0 = int(min(vertices[0][0],vertices[1][0]))
        y0 = int(min(vertices[0][1],vertices[1][1]))
        xf = int(max(vertices[0][0],vertices[1][0]))
        yf = int(max(vertices[0][1],vertices[1][1]))
        roi = {'x0':x0,'y0':y0,'xf':xf,'yf':yf,'roi':img[y0:yf,x0:xf]}
        return roi
    
    def findroi(self,img,tmplt):
        res = cv2.matchTemplate(img,tmplt,eval('cv2.TM_SQDIFF'))
        _,_,min_loc,_ = cv2.minMaxLoc(res)
        return min_loc
    
    def setDrop(self,img):
        #check: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
        boundingbox = self.getBox(img,'-- Bounding Box: Select two corners and close figure')
        vertices = self.getCoord(boundingbox['roi'],'-- Vertices: Select 4 points and close figure')[-4:]
        edge1 = self.getBox(boundingbox['roi'],'-- Left Edge: Select two corners around an edge and close figure')
        edge2 = self.getBox(boundingbox['roi'],'-- Right Edge: Select two corners around an edge and close figure')
        
        #find offsets with respect to the original (uncropped) frame 
        offset = {}
        offset['edge1'] = vertices[0][0] - self.findroi(boundingbox['roi'],edge1['roi'])[0] + boundingbox['x0']
        offset['edge2'] = vertices[3][0] - self.findroi(boundingbox['roi'],edge2['roi'])[0] + boundingbox['x0']
        drop = {'bbox':boundingbox,'vertices':vertices,'edge1':edge1,'edge2':edge2, 'offset':offset}
        return drop
    
    def volume(self,v,d):
        L = v[3] - v[0]
        tL = v[1] - v[0]
        tR = v[3] - v[3]
        r = d/float(2);
        return 0.001*(np.pi*L*r**2)-(np.pi*r**2)*(tL+tR)/float(2)+np.pi*(tL**3+tR**3)/float(6)
    
    def analyze(self,ndrops,c0=[1],refFrame = 1):
        sets = []
        allvertices = {}
        lengths = {}
        volume = {}
        concentration = {}
        #initialize drop
        for i in range(ndrops):
            print '-- Setting up drop {0}'.format(i+1)
            img = cv2.imread(self.imgs[refFrame-1])
            d = self.setDrop(img)
            sets.append(d)
            
        #run analysis on all frames
        for d in range(ndrops):
            lengths[d] = {}
            volume[d] = {}
            concentration[d] = {}
            allvertices[d] = {}
            print '-- Analyzing drop {0}'.format(d+1)
            drop = sets[d]
            bx0,by0,bxf,byf = drop['bbox']['x0'],drop['bbox']['y0'],drop['bbox']['xf'],drop['bbox']['yf']

            dv = (drop['vertices'][1][0]-drop['vertices'][0][0], #distance between first and second vertex
                  drop['vertices'][3][0]-drop['vertices'][2][0]) #distance between third and fourth vertex
            edge1 = drop['edge1']['roi']
            edge2 = drop['edge2']['roi']
            offset = drop['offset']
            vs= {}
            k = 1
            for imgpath in self.imgs[1:]:
                print os.path.split(imgpath)[1];
                img = cv2.imread(imgpath)
                im = img[by0:byf,bx0:bxf]
                x1 = self.findroi(im,edge1)[0] + offset['edge1']
                x2 = self.findroi(im,edge2)[0] + offset['edge2']
                allvertices[d][k] = {'1':x1, '2':x1+dv[0], '3': x2-dv[1], '4':x2}
                lengths[d][k] = x2-x1
                volume[d][k] = self.volume((x1,x1+dv[0],x2-dv[1],x2),100)
                if k == 1:
                    concentration[d][k] = c0[d]
                else:
                    concentration[d][k] = concentration[d][k-1]*volume[d][k-1]/float(volume[d][k])
                k+=1
        return allvertices,lengths,volume,concentration
