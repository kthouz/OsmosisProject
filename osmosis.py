import cv2
import os
import glob
import json
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scipyopt
import pprint


class Osmosis():
    """Osmosis experiment class"""
    def __init__(self,src_path, pxl2um, frame2sec, medium, frameIncr = 1, frameRateConversion = 1):
        print 'locating and loading '+src+' ...'
        self.src = src_path
        self.pxl2um = pxl2um
        self.frame2sec = frame2sec
        self.frameIncr = frameIncr
        self.frameRConv = frameRateConversion
        self.imgs = self.loadImages()[0]
        self.allimgs = self.loadImages()[1]
        self.ou_vertices = {}
        self.point = []
        self.medium = medium
        self.oudata = {'pxl2um':self.pxl2um,'frame2sec':self.frame2sec,'frameIncr':self.frameIncr,
                  'frameRateConversion':self.frameRConv,'medium':self.medium}
        print 'successful initiation'
        
        
    def loadImages(self):
        """This methods loads images to be analyzed using frameIncr"""
        try:
            allimgs = os.listdir(self.src)
            indices = np.arange(0,len(allimgs),self.frameIncr)
            imgs = [os.path.join(self.src,allimgs[i]) for i in indices]
        except:
            print "Directory not found, empty or frameIncr and number of images in directory are incompatible"
        return [imgs,allimgs]
    
    def __onclick__(self,click):
        """Method to detect a mouse click"""
        self.point.append((int(click.xdata),int(click.ydata)))
        return self.point
        
    def getCoord(self,img,s):
        """Method to get and record coordinates of the clicked position"""
        fig = plt.figure()
        plt.imshow(img)
        plt.title(s)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        return self.point
    
    def getBox(self,img,s):
        """Method to get and return selected box"""
        self.point = []
        vertices = self.getCoord(img,s)[-2:]
        x0 = int(min(vertices[0][0],vertices[1][0]))
        y0 = int(min(vertices[0][1],vertices[1][1]))
        xf = int(max(vertices[0][0],vertices[1][0]))
        yf = int(max(vertices[0][1],vertices[1][1]))
        roi = {'x0':x0,'y0':y0,'xf':xf,'yf':yf,'roi':img[y0:yf,x0:xf]}
        return roi
    
    def findroi(self,img,tmplt):
        """Match object and find the region of interest"""
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
        """Calculate volume of the drop"""
        L = (v[3] - v[0])*self.pxl2um
        tL = (v[1] - v[0])*self.pxl2um
        tR = (v[3] - v[2])*self.pxl2um
        r = 0.5*d
        p = np.pi
        return 0.001*((p*L*r**2)-(0.5*p*r**2*(tR+tL))+(p*((tL**3)+(tR**3))/6.0))
    
    def dumpjson(self,filename,data):
        """This function writes json files to disk"""
        with open(filename,'w') as outfile:
            json.dump(data,outfile)
        outfile.close()
    
    def analyze(self,ndrops,dropnames,c0=[1],refFrame = 1):
        self.dumpjson(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),'parameters'),self.oudata)
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
                #if k == 1:
                #    concentration[dn[d]][k] = c0[d]
                #else:
                #    concentration[dn[d]][k] = concentration[dn[d]][k-1]*volume[dn[d]][k-1]/float(volume[dn[d]][k])

                k+=1
            
            #save data to disc
            
            
            pd.DataFrame(allvertices).to_csv(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),dn[d] + '_vertices.csv'))
            #pd.DataFrame(lengths).to_csv(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),dn[d] + '_lengths.csv'))
            #pd.DataFrame(volume).to_csv(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),dn[d] + '_volume.csv'))
            #pd.DataFrame(concentration).to_csv(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),dn[d] + '_concentration.csv'))
        


        return allvertices,lengths,volume

    def saveSettings(self):
        self.dumpjson(os.path.join(os.path.abspath(os.path.join(self.src,os.pardir)),'settingsLog'),self.oudata)
        print "initial settings saved on disc"

    def loadResults(self,capillaryID,vType='drop'):
        src = os.path.dirname(self.src)
        
        allfiles = os.listdir(src)
        #files = []
        #for f in allfiles:
        #    if (vType+'_'+capillaryID+'_' in f ) and ('_vertices.csv' in f):
        #        files.append(f)
        if vType == 'drop':
            files = glob.glob(src+'\drop'+'_'+capillaryID+'*vertices.csv')
            files = [x for x in files if not 'gap' in x]
        if vType == 'gap':
            files = glob.glob(src+'\gap'+'_'+capillaryID+'*vertices.csv')
            files = [x for x in files if not 'drop' in x]
        k = 1
        for f in files:
            print f
            d = pd.read_csv(f)
            #d = pd.read_csv(os.path.join(src,f))
            d.columns = ['frame','v1','v2','v3','v4']
            d.index = [i for i in d['frame']]
            del d['frame']
            self.ou_vertices[re.sub('_vertices.csv','',os.path.basename(f))] = d
        #print "veritices dataset found: ";
        ks = self.ou_vertices.keys()
        ks.sort()
        #print ks
        
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

    def calculateVolumes(self,capID,dropName = None, single = True):
        """This function calculate the volume and concentration of the drop.
        capID: string - capillary ID,
        dropName: string - drop name,
        single: boolean - returns volume and concentration for a single droplet is true, otherwise it returns data for all available droplet data. Default to True"""
        d = 100 #capillary diameter
        self.ou_vertices = self.loadResults(capID,vType = 'drop')
        key = dropName
        if key == None:
            ks = self.ou_vertices.keys()
            ks.sort()
            key = ks[0]
            
        if single:
            #calculate the volume of a single drop
            #try:
            print key
            df = self.ou_vertices[key]
            df.plot()
            plt.title(key)
            plt.show()
                #volume
            vf = pd.DataFrame({key+'_vol_pl':pd.Series(data = [self.volume(df.loc[i].values,d) for i in df.index],index = df.index)})
            v0 = vf.loc[vf.index[0]][0]
                #concentration
            cf = pd.DataFrame({key+'_con_M':pd.Series(data = [self.concentration(vf.loc[i][0],v0,c0) for i in vf.index], index = vf.index)})
                #distance to neighboors
            gf = pd.DataFrame({key+'_d':pd.Series(data = self.d2neighboors(key), index = vf.index)})
            #except:
            #print 'querrying an empty ou_vertices dict. Try again'
            
        else:
            #calculate the volume of a all drops
            src = os.path.dirname(self.src)
            v = {}
            c = {}
            g = {}
            ks = self.ou_vertices.keys()
            ks.sort()
            for key in ks:
                #try:
                    #print key
                df = self.ou_vertices[key]
                    #volume
                v[key+'_vol_pl'] = pd.Series(data = [self.volume(df.loc[i].values,d) for i in df.index],index = df.index)
                pd.DataFrame(v[key+'_vol_pl'],columns = ['volume_pl']).to_csv(os.path.join(src,key+'_volume.csv'))
                    
                #except:
                #print 'querrying an empty ou_vertices dict. Try again'
            vf = pd.DataFrame(v)
                        
            #generate volume plot
            
            fig1 = plt.figure(figsize=(6,5))
            ax1 = fig1.add_subplot(111)
            (vf/1000.0).plot(x = vf.index*self.frame2sec*self.frameRConv*self.frameIncr/3600.0,
                    grid=False,ax=ax1,legend=False, colormap = 'jet')
            plt.legend([re.sub('drop_'+capID+'_','',key) for key in ks],
                       #ncol = int(math.ceil(len(ks)/10.)),
                       #loc =4,
                       fancybox = True, framealpha = 0.2,
                       fontsize = 10)
            plt.xlabel('Time (hrs)')
            plt.ylabel('Volume (nl)')
            plt.savefig(os.path.join(src,capID+'_volume.png'))
            plt.savefig(os.path.join(src,capID+'_volume.eps'))
            plt.show()

        return vf

    def calculateConcentrations(self,capID,initCs,dropName=None,single=True):
        """This function return the concentration given:
        capID: str, capillary ID,
        initCs: list of 2 elements, initial concentrations
        dropName: str, dropName
        single: boolean, default True"""
        src = os.path.dirname(self.src)
        vol = self.calculateVolumes(capID,dropName,single)
        c0 = initCs
        concentrations = {}
        if single == True:
            v = vol[key].dropna()
            if v[v.index[0]] > v[len(v)]:
                clist = [self.concentration(v[i],v[v.index[0]],c0[0]) for i in v.index]
            else:
                clist = [self.concentration(v[i],v[v.index[0]],c0[1]) for i in v.index]
            pd.DataFrame(clist, columns = ['concentration_M']).to_csv(os.path.join(src,key+'_concentration.csv'))
            concentrations[key] = pd.Series(data = clist, index = v.index)
        else:
            for key in vol.columns:
                v = vol[key].dropna()
                if v[v.index[0]] > v[len(v)]:
                    clist = [self.concentration(v[i],v[v.index[0]],c0[0]) for i in v.index]
                else:
                    clist = [self.concentration(v[i],v[v.index[0]],c0[1]) for i in v.index]
                
                pd.DataFrame(clist, columns = ['concentration_M']).to_csv(os.path.join(src,re.sub('_vol_pl','',key)+'_concentration.csv'))
                concentrations[key] = pd.Series(data = clist, index = v.index)

        #generate concentration plot
        cf = pd.DataFrame(concentrations)
        ks = vol.keys()
        fig2 = plt.figure(figsize=(6,5))
        ax2 = fig2.add_subplot(111)
        cf.plot(x = cf.index*self.frame2sec*self.frameRConv*self.frameIncr/3600.0,
                grid=False,ax=ax2,legend=False, colormap = 'jet')
        plt.legend([re.sub('_vol_pl','',re.sub('drop_'+capID+'_','',key)) for key in ks],
                #ncol = int(math.ceil(len(ks)/4.)),
                #loc = 4,
                fancybox = True, framealpha = 0.2,
                fontsize = 10)
        plt.xlabel('Time (hrs)')
        plt.ylabel('Concentration (M)')
        plt.savefig(os.path.join(src,capID+'_concentration.png'))
        plt.savefig(os.path.join(src,capID+'_concentration.eps'))
        plt.show()
        
        return pd.DataFrame(concentrations)

    def calculateGap(self,capID,dropName=None,vType='gap', single=True):
        """Calculate the gap distance between drops:
        capID: str, capillary ID
        dropName: str, name of the gap file
        vType: str boolean, can only take values gap and/or drop
        single: boolean"""
        src = os.path.dirname(self.src)
        vertices = self.loadResults(capID,vType)
        ks = vertices.keys()
        ks.sort()
        gapdistance = {}
        if dropName == None:
            key = ks[0]
        else:
            key = dropName
            
        if single == True:
            v = vertices[key]
            gap = (v['v4'] - v['v1'])*self.pxl2um
            gapdistance[key] = gap
            pd.DataFrame(gap,columns = ['gap_um']).to_csv(os.path.join(src,key+'.csv'))
        else:
            for key in ks:
                v = vertices[key]
                gap = (v['v4'] - v['v1'])*self.pxl2um
                gapdistance[key] = gap
                pd.DataFrame(gap,columns = ['gap_um']).to_csv(os.path.join(src,key+'.csv'))
        #plot gap distance
        cf = pd.DataFrame(gapdistance)
        fig3 = plt.figure(figsize=(6,5))
        ax3 = fig3.add_subplot(111)
        (cf/10.0).plot(x = cf.index*self.frame2sec*self.frameRConv*self.frameIncr/3600.0,
                grid=False,ax=ax3,legend=False, colormap = 'jet')
        plt.legend([re.sub('gap_'+capID+'_','',key) for key in ks],
                #ncol = int(math.ceil(len(ks)/4.)),
                #loc = 4,
                fancybox = True, framealpha = 0.2,
                fontsize = 10)
        plt.xlabel('Time (hrs)')
        plt.ylabel('distance between two drops (x10um)')
        plt.savefig(os.path.join(src,capID+'_gap.png'))
        plt.savefig(os.path.join(src,capID+'_gap.eps'))
        plt.show()
                          
        return pd.DataFrame(gapdistance)

    def findVelocitySquared(self,capID,dropName = None, single = True):
        src = os.path.dirname(self.src)
        self.ou_vertices = self.loadResults(capID,vType = 'drop')
        vertices = self.ou_vertices
        position = {}
        velocitySquared = {}
        if single == True:
            if dropName == None:
                dropName = vertices.keys()[0]
                vert = vertices[dropName]
            else:
                vert = vertices[dropName]
            p = ((vert['v4'] - vert['v1'])/2.0)*self.pxl2um
            v = pd.Series([(p[i]-p[i-1])/float(i-(i-1)) for i in p.index[1:]],p.index[1:])
            position[re.sub('drop_','',dropName)] = p
            velocitySquared[re.sub('drop_','',dropName)] = v**2
            pd.DataFrame(velocitySquared[re.sub('drop_','',key)]).to_csv(os.path.join(src,dropName+'_vSquared.csv'))
        else:
            for key in vertices.keys():
                vert = vertices[key]
                p = ((vert['v4'] - vert['v1'])/2.0)*self.pxl2um
                v = pd.Series([(p[i]-p[i-1])/float(i-(i-1)) for i in p.index[1:]],p.index[1:])
                position[re.sub('drop_','',key)] = p
                velocitySquared[re.sub('drop_','',key)] = v**2
                pd.DataFrame(velocitySquared[re.sub('drop_','',key)]).to_csv(os.path.join(src,key+'_vSquared.csv'))

        position = pd.DataFrame(position)
        velocitySquared = pd.DataFrame(velocitySquared)
        return position,velocitySquared

    def calculateRate(self,capID,dropSuffix,params):
        """This method calculate the rate of exchange
        capID: str, capillary ID
        dropSuffix: str, suffix of the format xx
        params: tuple or list, initial guesses for fitting"""
        #get data from file
        df = pd.read_csv(os.path.join(os.path.dirname(self.src),'drop_'+capID+'_'+dropSuffix+'_concentration.csv'))
        df[df.columns[0]] = df[df.columns[0]]*float(self.pxl2um)
        
        #fit
        fitpar,covpar = scipyopt.curve_fit(self.fit_func,df[df.columns[0]].values,df[df.columns[1]].values,p0=params)
        return fitpar, covpar

    def fit_func(self,xdata,a,b,x0,T):
        return a*np.exp(-(xdata-x0)/T)#-b

    def rsquared(self,x,y):
        pass

    def report(self,capID = '00'):
        """This method make and save a report about a drop as a json file. The json has followin keys
        * dropID: the drop identification string, AADDDDDDDDDCCNN. where AA stands for the setup used
        (00: New Microscope, 01: DIC SETUP, 02: MIS2, 03:MIS3), DDDDDDDDD is the experiment id (eg: 20150401b)
        CC is the capillary number and NN is the drop number
        * v0: initial volume
        * vf: final volumne
        * c0: initial concentration
        * cf: final concentration
        * medium: medium identification string, format ZZZXX. REA000: oil without surfactant, REA002: oil with 0.02%REA,
        REA020: oil with 0.2%REA, REA200: oil with 2%REA, REA400: oil with 4%REA, AIR000: no oil
        * g10: initial gap to first neighbor
        * g1f: final gap to first neighbor
        * g20: initial gap to second neighbor
        * g2f: final gap to second neighbor
        * T: exchan rate"""
        collection = {}
        doc = {}
        setups = {'New Microscope':'00','DICSETUP':'01','MIS2':'02','MIS3':'03'}
        usepath = self.src
        # read volume files
        vfiles = glob.glob(usepath+'\drop_'+capID+'*volume.csv')
        print(len(vfiles))
        # read concentration files
        cfiles = glob.glob(usepath+'\drop_'+capID+'*concentration.csv')
        print(len(cfiles))
        # read gap files
        vertfiles = glob.glob(usepath+'\gap_'+capID+'*.csv')
        gfiles = []
        for f in vertfiles:
            if not ('vertices' in f):
                gfiles.append(f)
        print(len(gfiles))

        for f in vfiles:
            doc['path'] = f
            data = pd.read_csv(f)
            #doc['volume'] = data.to_dict()
            doc['v0'] = data.volume_pl[:3].mean()
            doc['vf'] = data.volume_pl[-3:].mean()
            data = pd.read_csv(re.sub('volume','concentration',f))
            #doc['concentration'] = data.to_dict()
            doc['c0'] = data.concentration_M[:3].mean()
            doc['cf'] = data.concentration_M[-3:].mean()
            #ind = vfiles.index(f)
            ind = int(re.sub('_volume.csv','',f)[-2:])
            
            if ind == 1:
                data = pd.read_csv(os.path.join(os.path.dirname(f),'gap_'+capID+'_''{0:02}-{1:02}.csv'.format(ind,ind+1)))
                #data = pd.read_csv(gfiles[ind])
                #data['n2'] = data.to_dic()
                doc['n10'] = np.nan
                doc['n1f'] = np.nan
                doc['n20'] = math.ceil(data.gap_um[:3].mean())
                doc['n2f'] = math.ceil(data.gap_um[-3:].mean())
            if ind == len(gfiles):
                data = pd.read_csv(os.path.join(os.path.dirname(f),'gap_'+capID+'_''{0:02}-{1:02}.csv'.format(ind-1,ind)))
                #data = pd.read_csv(gfiles[ind-1])
                #data['n1'] = data.to_dic()
                doc['n10'] = math.ceil(data.gap_um[:3].mean())
                doc['n1f'] = math.ceil(data.gap_um[-3:].mean())
                doc['n20'] = np.nan
                doc['n2f'] = np.nan
            if ind>1 and ind<len(gfiles):
                data = pd.read_csv(os.path.join(os.path.dirname(f),'gap_'+capID+'_''{0:02}-{1:02}.csv'.format(ind-1,ind)))
                #data = pd.read_csv(gfiles[ind-1])
                doc['n10'] = math.ceil(data.gap_um[:3].mean())
                doc['n1f'] = math.ceil(data.gap_um[-3:].mean())
                data = pd.read_csv(os.path.join(os.path.dirname(f),'gap_'+capID+'_''{0:02}-{1:02}.csv'.format(ind,ind+1)))
                #data = pd.read_csv(gfiles[ind])
                doc['n20'] = math.ceil(data.gap_um[:3].mean())
                doc['n2f'] = math.ceil(data.gap_um[-3:].mean())
            #pprint.pprint(doc)
            collection[ind] = doc
            self.dumpjson(re.sub('_volume.csv','',f),doc)
            
        return collection, vfiles, cfiles, gfiles
        

if __name__ == '__main__':

    #src = r'F:\BackUp_Data\MIS3\20140226\transformed\c04_02'
    #src = r'F:\BackUp_Data\Old PIM\Camille\20140407\BBB2'
    #src = r'G:\BackUp_Data\MIS3\20150321\transformed\c07'
    #src = r'G:\BackUp_Data\MIS3\20140226\transformed'
    src = r'G:\BackUp_Data\MIS2\20150219\transformed'
    
    
    pxl2um =100/float(117)#100/float(100)#100/float(117)#100/float(103)#150/float(92)
    frame2sec = 82#60#15
    frameRateConversion = 8#25
    refFrame = 1#250
    frameIncr = 1
    med = 'REA200'
    test = Osmosis(src,pxl2um,frame2sec,med,frameIncr,frameRateConversion)

    #t = test.analyze(1,['gap_00_10-11','drop_00_19','drop_02_2','drop_02_17','drop_02_18'],[100,200,100,200,100],refFrame)
    
