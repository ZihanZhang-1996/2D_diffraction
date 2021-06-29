# Table atomic form factor, 
# Find at: http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
# or http://it.iucr.org/Cb/ch6o1v0001/.
import numpy as np
def AFF():
    # First column is the atomic number which is used as directory for function: Bragg_peaks() 
    F=np.array([[1,0.489918,20.6593,0.262003,7.74039,0.196767,49.5519,0.049879,2.20159,0.001305],
                [6,2.31,20.8439,1.02,10.2075,1.5886,0.5687,0.865,51.6512,0.2156],
                [7,12.2126,0.0057,3.1322,9.8933,2.0125,28.9975,1.1663,0.5826,-11.529],
                [53,20.2332,4.3579,18.997,0.3815,7.8069,29.5259,2.8868,84.9304,4.0714],
                [82,21.7886,1.3366,19.5682,0.488383,19.1406,6.7727,7.01107,23.8132,12.4734]])
    return F