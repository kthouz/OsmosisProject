import cv2
import os
import glob
import json
import re
import math
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
        L = (v[3] - v[0])*self.pxl2um
        tL = (v[1] - v[0])*self.pxl2um
        tR = (v[3] - v[2])*self.pxl2um
        r = 0.5*d
        p = np.pi
        return 0.001*((p*L*r**2)-(0.5*p*r**2*(tR+tL))+(p*((tL**3)+(tR**3))/6.0))
    
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

    def loadResults(self,capillaryID):
        src = os.path.dirname(self.src)
        allfiles = os.listdir(src)
        files = []
        for f in allfiles:
            if (capillaryID in f ) and ('_vertices.csv' in f):
                files.append(f)
        k = 1
        for f in files:
            d = pd.read_csv(os.path.join(src,f))
            d.columns = ['frame','v1','v2','v3','v4']
            d.index = [i for i in d['frame']]
            del d['frame']
            self.ou_vertices[re.sub('_vertices.csv','',os.path.basename(f))] = d
        print "veritices dataset found: ";
        print self.ou_vertices.keys()
        
        return self.ou_vertices

    def concentration(self,v,v0,c0):
        return c0*v0/float(v)

    def d2neighboors(self,dropname):
        ks = self.ou_vertices.keys()
        ks.sort()
        drop_index = ks.index(dropname)
        if drop_index == 0:
            d = (self.ou_vertices[ks[drop_index+1]]['v1']-self.ou_vertices[ks[drop_index]]['v4'])*self.pxl2um
        elif drop_index == len(ks)-1:
            d = (self.ou_vertices[ks[drop_index]]['v1'] - self.ou_vertices[ks[drop_index-1]]['v4'])*self.pxl2um
        else:
            d1 = (self.ou_vertices[ks[drop_index]]['v1'] - self.ou_vertices[ks[drop_index-1]]['v4'])*self.pxl2um
            d2 =(self.ou_vertices[ks[drop_index+1]]['v1']-self.ou_vertices[ks[drop_index]]['v4'])*self.pxl2um
            d = [(d1[i+1],d2[i+1]) for i in range(len(d1))]
        return d

    def calculateVolNConc(self,capID,c0,dropName = None, single = True):
        """This function calculate the volume and concentration of the drop.
        capID: string - capillary ID, c0: numeric or array or list - initial concentration,
        dropName: string - drop name, single: boolean - returns volume and concentration
        for a single droplet is true, otherwise it returns data for all available droplet data. Default to True"""
        d = 100 #capillary diameter
        self.ou_vertices = self.loadResults(capID)
        key = dropName
        if key == None:
            key = self.ou_vertices.keys()[0]
            
        if single:
            #calculate the volume of a single drop
            try:
                df = self.ou_vertices[key]
                df.plot()
                plt.show()
                #volume
                vf = pd.DataFrame({key+'_vol_pl':pd.Series(data = [self.volume(df.loc[i].values,d) for i in df.index],index = df.index)})
                v0 = vf.loc[vf.index[0]][0]
                #concentration
                cf = pd.DataFrame({key+'_con_M':pd.Series(data = [self.concentration(vf.loc[i][0],v0,c0) for i in vf.index], index = vf.index)})
                #distance to neighboors
                gf = pd.DataFrame({key+'_d':pd.Series(data = self.d2neighboors(key), index = vf.index)})
            except:
                print 'querrying an empty ou_vertices dict. Try again'
            
        else:
            #calculate the volume of a all drops
            v = {}
            c = {}
            g = {}
            ks = self.ou_vertices.keys()
            ks.sort()
            for key in ks:
                try:
                    df = self.ou_vertices[key]
                    #volume
                    v[key+'_vol_pl'] = pd.Series(data = [self.volume(df.loc[i].values,d) for i in df.index],index = df.index)
                    v0 = v[key+'_vol_pl'][v[key+'_vol_pl'].index[0]]
                    #concentration
                    c[key+'_vol_pl'] = pd.Series(data = [self.concentration(v[key+'_vol_pl'][i],v0,c0[ks.index(key)]
                                                                        ) for i in v[key+'_vol_pl'].index],index = v[key+'_vol_pl'].index)
                    #distance to neighboors
                    g[key+'_vol_pl'] = self.d2neighboors(key)
                    
                except:
                    print 'querrying an empty ou_vertices dict. Try again'
            vf = pd.DataFrame(v)
            cf = pd.DataFrame(c)
            gf = pd.DataFrame(g)
            
            #generate volume plot
            src = os.path.dirname(self.src)
            fig1 = plt.figure(figsize=(6,5))
            ax1 = fig1.add_subplot(111)
            (vf/1000.0).plot(x = vf.index*self.frame2sec*self.frameRConv*self.frameIncr/3600.0,
                    grid=False,ax=ax1,legend=False)
            plt.legend([i+1 for i in range(len(ks))],
                       ncol = int(math.ceil(len(ks)/4.)),
                       loc = 'upper right')
            plt.xlabel('Time (hrs)')
            plt.ylabel('Volume (nl)')
            plt.savefig(os.path.join(src,capID+'_volume.png'))
            plt.savefig(os.path.join(src,capID+'_volume.eps'))
            plt.show()

            #generate concentration plot
            fig2 = plt.figure(figsize=(6,5))
            ax2 = fig2.add_subplot(111)
            cf.plot(x = cf.index*self.frame2sec*self.frameRConv*self.frameIncr/3600.0,
                    grid=False,ax=ax2,legend=False)
            plt.legend([i+1 for i in range(len(ks))],
                       ncol = int(math.ceil(len(ks)/4.)),
                       loc = 'upper right')
            plt.xlabel('Time (hrs)')
            plt.ylabel('Concentration (M)')
            plt.savefig(os.path.join(src,capID+'_concentration.png'))
            plt.savefig(os.path.join(src,capID+'_concentration.eps'))
            plt.show()
            
            

        return vf,cf,gf
        

if __name__ == '__main__':
    
    src = r'path\\to\\directory\\Transformed'
    
    pxl2um = 100/float(103)#150/float(92)
    frame2sec = 60#15
    frameRateConversion = 25
    refFrame = 90
    frameIncr = 1
    test = TestClass(src,pxl2um,frame2sec,frameIncr,frameRateConversion)
    
    #for analysis uncomment this line
    #t = test.analyze(1,['_drop_00_06','drop_02_2','drop_02_17','drop_02_18'],[200,100,200,100,200,100],refFrame)
    
    #get equilibrium concentration and volume plots
    data = test.calculateVolNConc('00',[.2,.1,.2,.1,.2,.1,.2],single=False)

    
    
