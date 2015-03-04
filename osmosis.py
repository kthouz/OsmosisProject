import cv2
import os
import glob
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TestClass():
    def __init__(self,src_path, pxl2um, frame2sec, frameIncr = 1, frameRateConversion = 1):
        self.src = src_path
        self.pxl2um = pxl2um
        self.frame2sec = frame2sec
        self.frameIncr = frameIncr
        self.frameRConv = frameRateConversion
        self.imgs = self.loadImages()[0]
        self.allimgs = self.loadImages()[1]
        self.ou_vertices = {}
        self.point = []
        self.oudata = {'pxl2um':self.pxl2um,'frame2sec':self.frame2sec,'frameIncr':self.frameIncr,
                  'frameRateConversion':self.frameRConv}
        self.dumpjson(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),'parameters'),self.oudata)
        
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
        tR = v[3] - v[2]
        r = d/float(2);
        return 0.001*(np.pi*L*r**2)-(np.pi*r**2)*(tL+tR)*0.5+np.pi*((tL**3)+(tR**3))/float(6)
    
    def dumpjson(self,filename,data):
        """This function write json files to disk"""
        with open(filename,'w') as outfile:
            json.dump(data,outfile)
        outfile.close()
    
    def analyze(self,ndrops,dropnames,c0=[1],refFrame = 1):
        dn = dropnames
        sets = {}
        initSettings = {}
        allvertices = {}
        lengths = {}
        volume = {}
        concentration = {}
        #initialize drop
        for i in range(ndrops):
            print '-- Setting up drop {0}'.format(i+1)
            img = cv2.imread(self.imgs[refFrame-1])
            d = self.setDrop(img)
            sets[dn[i]] = d
        self.oudata['initSettings'] = sets
        
        #run analysis on all frames
        for d in range(ndrops):
            for ii in range(4):
                allvertices[dn[d]+'_'+str(ii+1)] = {}
            lengths[dn[d]] = {}
            volume[dn[d]] = {}
            concentration[dn[d]] = {}
            
            print '-- Analyzing drop {0}'.format(d+1)
            drop = sets[dn[d]]
            bx0,by0,bxf,byf = drop['bbox']['x0'],drop['bbox']['y0'],drop['bbox']['xf'],drop['bbox']['yf']

            dv = (drop['vertices'][1][0]-drop['vertices'][0][0], #distance between first and second vertex
                  drop['vertices'][3][0]-drop['vertices'][2][0]) #distance between third and fourth vertex
            edge1 = drop['edge1']['roi']
            edge2 = drop['edge2']['roi']
            offset = drop['offset']
            vs= {}
            k = 1
            for imgpath in self.imgs:
                #print os.path.split(imgpath)[1];
                img = cv2.imread(imgpath)
                im = img[by0:byf,bx0:bxf]
                x1 = self.findroi(im,edge1)[0] + offset['edge1']
                x2 = self.findroi(im,edge2)[0] + offset['edge2']
                allvertices[dn[d]+'_1'][k] = x1
                allvertices[dn[d]+'_2'][k] = x1+dv[0]
                allvertices[dn[d]+'_3'][k] = x2-dv[1]
                allvertices[dn[d]+'_4'][k] = x2
                lengths[dn[d]][k] = x2-x1
                volume[dn[d]][k] = (pxl2um**3)*self.volume((x1,x1+dv[0],x2-dv[1],x2),100/float(pxl2um))
                if k == 1:
                    concentration[dn[d]][k] = c0[d]
                else:
                    concentration[dn[d]][k] = concentration[dn[d]][k-1]*volume[dn[d]][k-1]/float(volume[dn[d]][k])
                k+=1
            
            #save data to disc
            
            
            pd.DataFrame(allvertices).to_csv(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),dn[d] + '_vertices.csv'))
            pd.DataFrame(lengths).to_csv(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),dn[d] + '_lengths.csv'))
            #pd.DataFrame(volume).to_csv(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),dn[d] + '_volume.csv'))
            #pd.DataFrame(concentration).to_csv(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),dn[d] + '_concentration.csv'))
        


        return allvertices,lengths,volume,concentration

    def saveSettings(self):
        self.dumpjson(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),'settingsLog'),self.oudata)
        print "initial settings saved on disc"

    def loadResults(self,dropID):
        src = os.path.dirname(self.src)
        allfiles = os.listdir(src)
        files = []
        for f in allfiles:
            if (dropID in f ) and ('_vertices.csv' in f):
                files.append(f)
        k = 1
        for f in files:
            d = pd.read_csv(os.path.join(src,f))
            d.index = d['_index']
            del d['_index']
            self.ou_vertices[re.sub('_vertices.csv','',os.path.basename(f))] = d
        print "veritices dataset found: ";
        print self.ou_vertices.keys()
        return self.ou_vertices

if __name__ == '__main__':
    #src = r'/media/camille/Seagate Backup Plus Drive/BackUp_Data/DICSETUP/20150219/Transformed/1-5' #1
    #src = r'C:\\Users\\camille\\Box Sync\\owncloud Files\\PhD\\DoD\\20150113\\Transformed\\1' #2
    src = r'F:\\BackUp_Data\\MIS3\\20140226\\transformed\\c03_00'
    
    pxl2um = 100/float(103)#150/float(92)
    frame2sec = 60#15
    frameRateConversion = 25
    refFrame = 90
    frameIncr = 1
    test = TestClass(src,pxl2um,frame2sec,frameIncr,frameRateConversion)

    #t = test.analyze(2,['_drop_02_1','drop_02_2','drop_02_17','drop_02_18'],[100,200,100,200,100],refFrame)
    
