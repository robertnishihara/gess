from IPython import parallel
import numpy

import gess

rc = parallel.Client(packer='pickle') # for running on starcluster on
                                      # EC2. This line may have to be
                                      # modified when running on other
                                      # clusters
dview = rc[:]
print "using " + str(len(rc.ids)) + " engines"

with dview.sync_imports():
    import numpy
    import os

working_dir = os.getcwd()
dview.execute("os.chdir('" + working_dir + "')")
dview.execute("os.environ['MKL_NUM_THREADS']='1'") # prevent numpy from multithreading

dim = 10
num_cores = len(rc.ids)
iters = 1000
burnin = 1000
thinning = 1
repeats = 100
chains = num_cores

# a simple pdf
def logf(x):
    return -.5 * numpy.dot(x, x)

starts = numpy.random.normal(loc=numpy.zeros(dim), scale=1, size=(2*chains,dim))
samples, calls = gess.parallel_gess(chains, iters, burnin, thinning, starts, logf, repeats, dview)
