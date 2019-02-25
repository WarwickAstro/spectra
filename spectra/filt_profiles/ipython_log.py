#log# Automatic Logger file. *** THIS MUST BE THE FIRST LINE ***
#log# DO NOT CHANGE THIS LINE OR THE TWO BELOW -* coding: UTF-8 *-
#log# opts = Struct({'__allownew': True, 'logfile': 'ipython_log.py', 'pylab': 1})
#log# args = []
#log# It is safe to make manual edits below here.
#log#-----------------------------------------------------------------------
x, y = np.loadtxt("SLOAN_SDSS.g.dat",unpack
=True)
plot(x,y)
