import numpy as np
import AFF
def read_poscar(address):
    f=open(address)
    ind=0
    for x in f:
        ind+=1
        if ind==3:
            x=x.lstrip()
            x=x.rstrip()
            x=x.split()
            x=np.array(x)
            p=x.astype(float)
            a1=p
        if ind==4:
            x=x.lstrip()
            x=x.rstrip()
            x=x.split()
            x=np.array(x)
            p=x.astype(float)
            a2=p
        if ind==5:
            x=x.lstrip()
            x=x.rstrip()
            x=x.split()
            x=np.array(x)
            p=x.astype(float)
            a3=p
        if ind==6:
            x=x.lstrip()
            x=x.rstrip()
            x=x.split()
            x=np.array(x)
            z=x.astype(int)
        if ind==7:
            x=x.lstrip()
            x=x.rstrip()
            x=x.split()
            x=np.array(x)
            z1=x.astype(int)
            temp=np.sum(z1)
            position=np.zeros((temp,4))
        if ind>8:
            x=x.lstrip()
            x=x.rstrip()
            x=x.split("         ")
            x=np.array(x)
            p=x.astype(float)
            position[ind-9,1]=p[0]
            position[ind-9,2]=p[1]
            position[ind-9,3]=p[2]
    f.close()
    ind=0
    iii=0
    for ii in z1:
        position[iii:iii+ii+1,0]=z[ind]
        iii=iii+ii
        ind=ind+1
    return a1,a2,a3,position

# Bragg_peaks function calculate the position of Bragg peaks in reciprocal space 
# using lattice parameters and position of atoms read from the POSCAR file.
# Two rotation angles respect to x and y axis are added to adjust the orientation of the single crystal.
def Bragg_peaks(a1,a2,a3,positions,thetax,thetay,hkl_dimension):
    # Lattice parameters M matrix in cartesian coordinate(angstrom)
    M=[a1,a2,a3]
    M=np.asarray(M)
    # Rotation Matrix respect to X axis, rotation angle = thetax
    Rx=np.array([[1,0,0],[0,np.cos(thetax),-np.sin(thetax)],[0,np.sin(thetax),np.cos(thetax)]])
    # Rotation Matrix respect to Y axis, rotation angle = thetay
    Ry=np.array([[np.cos(thetay),0,-np.sin(thetay)],[0,1,0],[np.sin(thetay),0,np.cos(thetay)]])

    # Rotation of the sample
    M=np.matmul(M, Rx)
    M=np.matmul(M, Ry)
    
    # New lattice parameter
    aa1=M[0,:]
    aa2=M[1,:]
    aa3=M[2,:]

    # reciprocal lattice
    volume=np.matmul(aa3,np.cross(aa1,aa2))
    b1=2*np.pi*np.cross(aa2,aa3)/volume
    b2=2*np.pi*np.cross(aa3,aa1)/volume
    b3=2*np.pi*np.cross(aa1,aa2)/volume

    # grid for Miller index
    i=np.linspace(-hkl_dimension,hkl_dimension,2*hkl_dimension+1)
    H,K,L=np.meshgrid(i,i,i)
    
    # The position of Bragg peaks in reciprocal space
    G1=H*b1[0]+K*b2[0]+L*b3[0]
    G2=H*b1[1]+K*b2[1]+L*b3[1]
    G3=H*b1[2]+K*b2[2]+L*b3[2]
    
    ss=np.size(positions)/4
    ss=int(ss)
    
    # load atomic form factor table
    AF=AFF.AFF()
    
    # calculate the atomic form factor
    ii=np.linspace(0,ss-1,ss)
    ii=ii.astype(int)
    q2=G1*G1+G2*G2+G3*G3
    F=0
    for j in ii:
        x = np.searchsorted(AF[:,0],positions[j,0])
        fq=0
        # first formula at http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
        fq=fq+AF[x,1]*np.exp(-AF[x,2]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,3]*np.exp(-AF[x,4]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,5]*np.exp(-AF[x,6]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,7]*np.exp(-AF[x,8]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,9]
        # position of atom in real space cartesian coordinate(angstrom)
        RR=positions[j,1]*aa1+positions[j,2]*aa2+positions[j,3]*aa3
        F=F+fq*np.exp(1j*(G1*RR[0]+G2*RR[1]+G3*RR[2]))
    F=np.abs(F)
    F=pow(F,2)
    Bpeaks=np.concatenate((G1,G2,G3,F), axis=0)
    return Bpeaks

# Intensity calculation
def intensity(gridx,gridz,Bpeaks,sigma1,sigma2,sigma3,hkl_dimension):
    iMiller=hkl_dimension*2+1
    G1=Bpeaks[0:iMiller,:,:]+np.finfo(float).eps
    G2=Bpeaks[iMiller:2*iMiller,:,:]+np.finfo(float).eps
    G3=Bpeaks[2*iMiller:3*iMiller,:,:]+np.finfo(float).eps
    F=Bpeaks[3*iMiller:4*iMiller,:,:]
    
    Eye=np.ones((iMiller,iMiller,iMiller))
    # The positions(r0,theta0,phi0) of Bragg peaks in spherical coordinates.
    theta0=np.pi/2-np.arctan(G3/np.sqrt(pow(G2,2)+pow(G1,2)))
    phi0=np.ones((iMiller,iMiller,iMiller))
    i=np.arange(iMiller)
    for k1 in i:
        for k2 in i:
            for k3 in i:
                if G1[k1,k2,k3]>0:
                    phi0[k1,k2,k3]=np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                else:
                    phi0[k1,k2,k3]=np.pi+np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                if abs(G2[k1,k2,k3])<0.2:
                    if abs(G1[k1,k2,k3])<0.2:
                        phi0[k1,k2,k3]=0
                    
    r0=np.sqrt(pow(G1,2)+pow(G2,2)+pow(G3,2))

    # The positions(r,theta,phi) of image plane in spherical coordinates.
    ix,iy=gridx.shape
    I0=np.ones((ix,iy))
    ix=np.arange(ix)
    iy=np.arange(iy)
    for x in ix:
        for y in iy:
            theta=np.pi/2-np.arctan(gridz[x,y]/abs(gridx[x,y]))
            r=np.sqrt(pow(gridx[x,y],2)+pow(gridz[x,y],2))
            if gridx[x,y]>0:
                phi=0
            else:
                phi=np.pi
            phi=phi*Eye
            phid=abs(phi-phi0)
            phid=abs(abs(phid-np.pi)-np.pi)
            I1=np.exp(-0.5*pow(theta*Eye-theta0,2)/sigma1/sigma1)
            I2=np.exp(-0.5*phid*phid/sigma2/sigma2)
            I3=np.exp(-0.5*pow(r*Eye-r0,2)/sigma3/sigma3)
            Intensity=I1*I2*I3*F
            I0[x,y]=np.sum(Intensity)
    return I0