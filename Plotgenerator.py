import pandas as pd
import numpy as np
import os
import json
import glob
import re
import matplotlib.pyplot as plt

class Plotgenerator():
    """This class generates plots of concentration, volume, velocity and gap between drops in the osmosis experiments.
    Use this class after finishing initial analyzis. See osmosis.py"""
    def __init__(self,src):
        """Initialization"""
        self.src = src
        self.parameters = self.importSettings()
        self.files = self.loadAllCsv()
        self.files.sort()
        
    def importSettings(self):
        jsonfile = json.load(open(os.path.join(self.src,'parameters')))
        return jsonfile

    def loadAllCsv(self):
        paths = glob.glob(os.path.join(self.src+'\*.csv'))
        files = [os.path.basename(f) for f in paths if ('vertices' not in f)]
        files = [f for f in files if (('gap_' not in f) or ('_concentration.csv' not in f))]
        files = [f for f in files if (('gap_' not in f) or ('_volume.csv' not in f))]
        return files

    def plotConcentration(self,capID,dropStr):
        data = {}
        for s in dropStr:
            cf = pd.read_csv(os.path.join(self.src,'drop_'+capID+'_'+s+'_concentration.csv'))
            cf.index = cf[cf.columns[0]]
            del cf[cf.columns[0]]
            cf = cf.to_dict()
            cf = cf[cf.keys()[0]]
            data[re.sub('drop_'+capID,'',s)] = pd.Series(data = cf.values(),
                                                   index = cf.keys())

        data = pd.DataFrame(data)
        self.genPlot(data,dropStr,'Time (s)','Concentratoin (M)')
        return data

    def plotVolume(self,capID,dropStr):
        data = {}
        for s in dropStr:
            cf = pd.read_csv(os.path.join(self.src,'drop_'+capID+'_'+s+'_volume.csv'))
            cf.index = cf[cf.columns[0]]
            del cf[cf.columns[0]]
            cf = cf.to_dict()
            cf = cf[cf.keys()[0]]
            data[re.sub('drop_'+capID,'',s)] = pd.Series(data = cf.values(),
                                                   index = cf.keys())

        data = pd.DataFrame(data)
        self.genPlot((data/1000),dropStr,'Time (s)','volume (nl)')
        return data
    
    def plotVSquared(self,capID,dropStr):
        data = {}
        for s in dropStr:
            cf = pd.read_csv(os.path.join(self.src,'drop_'+capID+'_'+s+'_vSquared.csv'))
            cf.index = cf[cf.columns[0]]
            del cf[cf.columns[0]]
            cf = cf.to_dict()
            cf = cf[cf.keys()[0]]
            data[re.sub('drop_'+capID,'',s)] = pd.Series(data = cf.values(),
                                                   index = cf.keys())

        data = pd.DataFrame(data)
        self.genPlot(data,dropStr,'Time (s)','velocity squared ((um/s)^2)')
        return data

    def plotGap(self,capID,gapStr):
        data = {}
        for s in gapStr:
            cf = pd.read_csv(os.path.join(self.src,'gap_'+capID+'_'+s+'.csv'))
            cf.index = cf[cf.columns[0]]
            del cf[cf.columns[0]]
            cf = cf.to_dict()
            cf = cf[cf.keys()[0]]
            data[re.sub('gap_'+capID,'',s)] = pd.Series(data = cf.values(),
                                                   index = cf.keys())

        data = pd.DataFrame(data)
        self.genPlot((data/10),gapStr,'Time (s)','Distance between drops (x10um)')
        return data


    def genPlot(self,data,leg,xlab,ylab):
        pxl2um = self.parameters['pxl2um']
        frameIncr = self.parameters['frameIncr']
        frameRConv = self.parameters ['frameRateConversion']
        frame2sec = self.parameters['frame2sec']
        
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111)
        data.plot(x =data.index*frame2sec*frameRConv*frameIncr/3600.0,
                grid=False,ax=ax,legend=False, colormap = 'jet')
        plt.legend(leg,
                fancybox = True, framealpha = 0.2,
                fontsize = 10)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()


if __name__ == '__main__':
    src = r'F:\BackUp_Data\MIS3\20140226\transformed'
    test = Plotgenerator(src)
