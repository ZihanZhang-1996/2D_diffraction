import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
class GIWAXS:
    def __init__(self,data,qxy,qz,name,dirr):
        self.name=name
        self.data=data
        self.qxy=qxy
        self.qz=qz
        self.qxymax=qxy[1]
        self.qxymin=qxy[0]        
        self.qzmax=qz[1]
        self.qzmin=qz[0]
        self.yp,self.xp=self.data.shape
        self.dirr=dirr
        
    def rename(self,name):
        self.name=name
        
    def imshow(self):
        lb = np.nanpercentile(self.data, 10)
        ub = np.nanpercentile(self.data, 99.9)
        print('yes')
        fig,ax=plt.subplots(figsize=(7,7))
        ax.imshow(self.data, interpolation='nearest', cmap=cm.jet,
               origin='lower', extent=[self.qxymin, self.qxymax, self.qzmin, self.qzmax],
               vmax=ub, vmin=lb)
        ax.set_xlabel('q$_{xy}$',fontsize=16)
        ax.set_ylabel('q$_{z}$',fontsize=16)
        ax.set_title(self.name,fontsize=16)
        
    def cut(self,qxy,qz,*flip):
        for fl in flip:
            if fl == 'f':
                self.data=np.flip(self.data)
        xL=math.floor((qxy[1]-qxy[0] )/(self.qxymax-self.qxymin)*self.xp)
        yL=math.floor((qz[1]-qz[0])/(self.qzmax-self.qzmin)*self.yp)
        xs=math.floor((qxy[0] -self.qxymin)/(self.qxymax-self.qxymin)*self.xp)
        ys=math.floor((qz[0]-self.qzmin)/(self.qzmax-self.qzmin)*self.yp)
        self.cut_qxymax=qxy[1]
        self.cut_qxymin=qxy[0]        
        self.cut_qzmax=qz[1]
        self.cut_qzmin=qz[0]
        self.cut_data=self.data[ys:ys+yL,xs:xs+xL]
        self.cut_yp,self.cut_xp=self.cut_data.shape
        for fl in flip:
            if fl == 'f':
                self.data=np.flip(self.data)
                
    def cut_imshow(self):
        lb = np.nanpercentile(self.cut_data, 10)
        ub = np.nanpercentile(self.cut_data, 99)
        fig,ax=plt.subplots(figsize=(7,7))
        ax.imshow(self.cut_data, interpolation='nearest', cmap=cm.jet,
               origin='lower', extent=[self.cut_qxymin, self.cut_qxymax, self.cut_qzmin, self.cut_qzmax],
               vmax=ub, vmin=lb)
        ax.set_xlabel('q$_{xy}$(1/A)',fontsize=16)
        ax.set_ylabel('q$_{z}(1/A)$',fontsize=16)
        ax.set_title(self.name,fontsize=16)
        
    def qzint(self,qxy,qz):
        xL=math.floor((qxy[1]-qxy[0] )/(self.cut_qxymax-self.cut_qxymin)*self.cut_xp)
        yL=math.floor((qz[1]-qz[0])/(self.cut_qzmax-self.cut_qzmin)*self.cut_yp)
        xs=math.floor((qxy[0] -self.cut_qxymin)/(self.cut_qxymax-self.cut_qxymin)*self.cut_xp)
        ys=math.floor((qz[0]-self.cut_qzmin)/(self.cut_qzmax-self.cut_qzmin)*self.cut_yp)
        self.qzint_qxymax=qxy[1]
        self.qzint_qxymin=qxy[0]        
        self.qzint_qzmax=qz[1]
        self.qzint_qzmin=qz[0]
        self.qzint_data=self.cut_data[ys:ys+yL,xs:xs+xL]
        self.qzint_I=sum(np.transpose(self.qzint_data))
        self.qzint_mask=np.zeros([self.cut_yp,self.cut_xp])
        self.qzint_mask[ys:ys+yL,xs:xs+xL]=self.cut_data[ys:ys+yL,xs:xs+xL]
        return np.linspace(self.qzint_qzmin,self.qzint_qzmax,self.qzint_I.shape[0]),self.qzint_I
        
    def qzint_imshow(self):
        lb = np.nanpercentile(self.qzint_data, 10)
        ub = np.nanpercentile(self.qzint_data, 99)
        fig,ax=plt.subplots(3,1,figsize=(7,21))
        ax[0].imshow(self.qzint_mask, interpolation='nearest', cmap=cm.jet,
               origin='lower', extent=[self.cut_qxymin, self.cut_qxymax, self.cut_qzmin, self.cut_qzmax],
               vmax=ub, vmin=lb)
        ax[0].set_xlabel('q$_{xy}$',fontsize=16)
        ax[0].set_ylabel('q$_{z}$',fontsize=16)
        ax[0].set_title(self.name+' qzint Mask',fontsize=16)
        
        ax[1].imshow(self.cut_data, interpolation='nearest', cmap=cm.jet,
               origin='lower', extent=[self.cut_qxymin, self.cut_qxymax, self.cut_qzmin, self.cut_qzmax],
               vmax=ub, vmin=lb)
        ax[1].set_xlabel('q$_{xy}$',fontsize=16)
        ax[1].set_ylabel('q$_{z}$',fontsize=16)
        ax[1].set_title(self.name+' original',fontsize=16)
        
        ax[2].plot(np.linspace(self.qzint_qzmin,self.qzint_qzmax,self.qzint_I.shape[0]),self.qzint_I)
        ax[2].set_xlabel('q$_{z}$',fontsize=16)
        ax[2].set_ylabel('Intensity(a.u.)',fontsize=16)
        ax[2].set_title(self.name+' I vs q$_z$',fontsize=16)
        
    def qxyint(self,qxy,qz):
        xL=math.floor((qxy[1]-qxy[0] )/(self.cut_qxymax-self.cut_qxymin)*self.cut_xp)
        yL=math.floor((qz[1]-qz[0])/(self.cut_qzmax-self.cut_qzmin)*self.cut_yp)
        xs=math.floor((qxy[0] -self.cut_qxymin)/(self.cut_qxymax-self.cut_qxymin)*self.cut_xp)
        ys=math.floor((qz[0]-self.cut_qzmin)/(self.cut_qzmax-self.cut_qzmin)*self.cut_yp)
        self.qxyint_qxymax=qxy[1]
        self.qxyint_qxymin=qxy[0]        
        self.qxyint_qzmax=qz[1]
        self.qxyint_qzmin=qz[0]
        self.qxyint_data=self.cut_data[ys:ys+yL,xs:xs+xL]
        self.qxyint_I=sum(self.qxyint_data)
        self.qxyint_mask=np.zeros([self.cut_yp,self.cut_xp])
        self.qxyint_mask[ys:ys+yL,xs:xs+xL]=self.cut_data[ys:ys+yL,xs:xs+xL]
        
    def qxyint_imshow(self):
        lb = np.nanpercentile(self.qxyint_data, 10)
        ub = np.nanpercentile(self.qxyint_data, 99)
        fig,ax=plt.subplots(3,1,figsize=(7,21))
        ax[0].imshow(self.qxyint_mask, interpolation='nearest', cmap=cm.jet,
               origin='lower', extent=[self.cut_qxymin, self.cut_qxymax, self.cut_qzmin, self.cut_qzmax],
               vmax=ub, vmin=lb)
        ax[0].set_xlabel('q$_{xy}$',fontsize=16)
        ax[0].set_ylabel('q$_{z}$',fontsize=16)
        ax[0].set_title(self.name+' qxyint Mask',fontsize=16)
        
        ax[1].imshow(self.cut_data, interpolation='nearest', cmap=cm.jet,
               origin='lower', extent=[self.cut_qxymin, self.cut_qxymax, self.cut_qzmin, self.cut_qzmax],
               vmax=ub, vmin=lb)
        ax[1].set_xlabel('q$_{xy}$',fontsize=16)
        ax[1].set_ylabel('q$_{z}$',fontsize=16)
        ax[1].set_title(self.name+' original',fontsize=16)
        
        ax[2].plot(np.linspace(self.qxyint_qxymin,self.qxyint_qxymax,self.qxyint_I.shape[0]),self.qxyint_I)
        ax[2].set_xlabel('q$_{xy}(1/A)$',fontsize=16)
        ax[2].set_ylabel('Intensity(a.u.)',fontsize=16)
        ax[2].set_title(self.name+' I vs q$_z$',fontsize=16)
        
    def aglint(self,angle,qrange,qp,smooth=False,**kwargs):
        self.aglint_mask=np.zeros([self.cut_yp,self.cut_xp])
        self.aglint_qp=qp
        self.aglint_qrange=qrange
        self.aglint_agl=angle
        xline=np.linspace(1,self.cut_xp-1,self.cut_xp-1)
        xline=xline.astype(int)
        yline=np.linspace(1,self.cut_yp-1,self.cut_yp-1)
        yline=yline.astype(int)
        self.aglint_I=np.zeros(qp)
        area=np.zeros(qp)
        self.aglint_q=np.linspace(self.aglint_qrange[0],self.aglint_qrange[1],self.aglint_I.shape[0])
        for i in xline:
            for j in yline:
                a=(self.cut_qxymax-self.cut_qxymin)/self.cut_xp*i+self.cut_qxymin
                b=(self.cut_qzmax-self.cut_qzmin)/self.cut_yp*j+self.cut_qzmin
                q=np.sqrt(a*a+b*b)
                angle1=np.arccos(a/q)
                qr=qrange[1]-qrange[0]
                qi=math.floor((q-qrange[0])/(qr/qp))
                angle1=angle1/np.pi*180-90
                if angle1>angle[0]:
                    if angle1<angle[1]:
                        if q>qrange[0]:
                            if q<qrange[1]:
#                                 if np.not_equal(self.cut_data[j,i],0):
                                self.aglint_I[qi]= self.aglint_I[qi]+self.cut_data[j,i]
                                self.aglint_mask[j,i]=self.aglint_mask[j,i]+self.cut_data[j,i]
                                area[qi]=area[qi]+1
        if smooth:
            N_smth=kwargs['Nsmooth']
            conv=np.ones(N_smth)/N_smth
            convx=np.linspace(0,N_smth,N_smth)
            conv1=np.exp(-convx*convx/N_smth/N_smth/2)
            conv1=conv1/np.sum(conv1)
            plt.plot(convx,conv1)
            self.aglint_I=np.convolve(conv1,self.aglint_I,'same')
        self.aglint_I=self.aglint_I/area
        return self.aglint_q,self.aglint_I
                                
    def aglint_imshow(self,save=False):
        lb = np.nanpercentile(self.cut_data, 10)
        ub = np.nanpercentile(self.cut_data, 99)
        fig,ax=plt.subplots(3,1,figsize=(7,21))
        ax[0].imshow(self.aglint_mask, interpolation='nearest', cmap=cm.jet,
               origin='lower', extent=[self.cut_qxymin, self.cut_qxymax, self.cut_qzmin, self.cut_qzmax],
               vmax=ub, vmin=lb)
        ax[0].set_xlabel('q$_{xy}$',fontsize=16)
        ax[0].set_ylabel('q$_{z}$',fontsize=16)
        ax[0].set_title(self.name+' aglint Mask',fontsize=16)
        
        ax[1].imshow(self.cut_data, interpolation='nearest', cmap=cm.jet,
               origin='lower', extent=[self.cut_qxymin, self.cut_qxymax, self.cut_qzmin, self.cut_qzmax],
               vmax=ub, vmin=lb)
        ax[1].set_xlabel('q$_{xy}$(1/A)',fontsize=16)
        ax[1].set_ylabel('q$_{z}$(1/A)',fontsize=16)
        ax[1].set_title(self.name+' original',fontsize=16)
        
        ax[2].plot(self.aglint_q,self.aglint_I)
        ax[2].set_xlabel('q(1/A)',fontsize=16)
        ax[2].set_ylabel('Intensity(a.u.)',fontsize=16)
        ax[2].set_title(self.name+' I vs q$_z$',fontsize=16)
        if save:
            name0=self.dirr+'py_data/'+self.name+'_aglint_'+ str(self.aglint_agl[0])+'_'+str(self.aglint_agl[1])+'_'+str(self.aglint_qrange[0])+'_'+str(self.aglint_qrange[1])+'_'+str(self.aglint_qp)
            dave=np.array([self.aglint_q,self.aglint_I,np.ones(self.aglint_q.shape[0])])
            plt.savefig(name0+'.pdf')
            df=pd.DataFrame([self.aglint_q,self.aglint_I])
            df.to_csv(name0+'.csv')
            name1=name0.replace( "/","\\" )
#                 3-col data with fake error; this can be transfer to DAVE for further data analysis
            dave=np.transpose(dave)
            np.savetxt(name1+'.txt',dave)
        return self.aglint_q,self.aglint_I