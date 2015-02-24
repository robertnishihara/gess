# Gess

Generalized Elliptical Slice Sampling: An MCMC algorithm for sampling continuous probability distributions in parallel

Suppose you want to run MCMC and you have access to a lot of cores, what is the best that you can do? Of course you could run a number of independent Markov chains in parallel, but it would be nice if there was a way to share information between Markov chains to speed up the process of sampling. Gess accomplishes this by using the current states of the parallel Markov chains to build an approximation to the target distribution. This approximation is used to factor the target distribution in a form that enables us to use [Elliptical Slice Sampling](http://arxiv.org/pdf/1001.0175v2.pdf) to update each Markov chain in parallel. See the [paper](http://arxiv.org/pdf/1210.7477v2.pdf) for a description of the algorithm.

## Using IPython for Parallelism

Gess is built in Python, and it uses the [IPython.parallel module](http://ipython.org/ipython-doc/dev/parallel/) to scale MCMC over large clusters.

### Clearing Caches

Depending on the version of IPython that you run, you may find that it tries to cache everything, which will cause your experiment to crash. The various caches can be cleared as follows:

    def clear_cache(rc, dview):
        rc.purge_results('all') #clears controller
        rc.results.clear()
        rc.metadata.clear()
        dview.results.clear()
        assert not rc.outstanding, "don't clear history when tasks are outstanding"
        rc.history = []
        dview.history = []

### Controlling Parallelism

If you are using IPython in conjunction with NumPy, you may find the load on your cluster growing unreasonably (perhaps to the point that the scheduler starts suspending jobs). This problem likely has something to do with IPython and NumPy both trying to make use of parallelism. IPython manages parallelism by creating one engine (each engine is a separate Python instance) for each available core. Then the IPython controller distributes computation to the engines and collects back the results. On the other hand, some NumPy functions try to fork a large number of threads (by calling functions from the Intel Math Kernel Library). This can be addressed by setting the appropriate environment variables on each engine:

    dview.execute("os.chdir('" + os.getcwd() + "')")
    dview.execute("os.environ['MKL_NUM_THREADS']='1'") # prevent numpy from multithreading


## Building the Approximation

Gess shares information between the parallel Markov chains by using their current states to build an approximation to the target distribution. As described in the paper, the user has some flexibility in what class of approximation to use. Anything that vaguely resembles a multivariate Gaussian distribution will work (for instance, a multivariate t distribution or a mixture of Gaussians will work). The code provided in `gess.py` uses a multivariate t approximation. The function `fit_mvstud.py` chooses the parameters of the t distribution using the algorithm described in this [paper](http://www3.stat.sinica.edu.tw/statistica/oldpdf/a5n12.pdf).

However, the user's ability to build a good approximation depends on how the number of Markov chains compares to the dimension of the target distribution. If the number of Markov chains used for building the approximation is sufficiently in excess of the dimension of the distribution, then `fit_mvstud.py` will work perfectly. If not, a different approximation scheme may have to be used, or the current states of the Markov chain will have to padded with auxiliary data (perhaps drawn from a spherical Gaussian) in order to build the approximation. With a very large number of Markov chains, a mixture of Gaussians approximation will likely provide the best performance.

## Usage

Sample usage is shown in the script `example.py`, which can be run on an EC2 cluster with sufficiently many cores. If you intend to use EC2, you should read about using [IPython on starcluster](http://star.mit.edu/cluster/docs/0.93.3/plugins/ipython.html). If you intend to use Gess on a different cluster, you will probably have to make slight adjustments.