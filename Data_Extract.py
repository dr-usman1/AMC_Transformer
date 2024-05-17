import numpy as np
import pickle as cPickle
# import cPickle
import sys
import cmath

with open("RML2016.10a_dict.pkl", 'rb') as f:
    Xd = cPickle.load(f, encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
for mod in mods:
    print ("Modulation Type ", mod)
    for snr in snrs:
        print ('SNR', snr)
        X = Xd[(mod,snr)]
        print ("Number of files", X.shape[0])
        for ind in range(X.shape[0]):
            # for ind in range(100):
            # print ” File “, ind
            Y = X[ind]
            Z = np.zeros((1, Y.shape[1]),dtype=complex)
        for c in range(Y.shape[1]):
            Z[0, c] = complex(Y[0, c], Y[1, c])
        # print np.abs(Z[0,c]), Y[0, c], Y[1, c]

        # Create a filename string from the mod type and the SNR.

        fn ='rml_' + mod + '_' + str(snr) + '_' + str(ind) + '.tim'
        # print(fn)

        # Open a file for writing using the created string

        fn_fid = open (fn, "w")
        dummy = 'filname'

        # Write the data in ASCII CMS format.dummy = str(2) + ‘\n’
        fn_fid.write(dummy)
        dummy = str(Y.shape[1]) + '\n'
        fn_fid.write(dummy)
        for c in range(Y.shape[1]):

            dummy = str(Z[0, c].real) + ''
            fn_fid.write(dummy)
            dummy = str(Z[0, c].imag) + '\n'
            fn_fid.write(dummy)

            # Close the file.

            fn_fid.close()