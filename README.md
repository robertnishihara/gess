# Gess

Generalized Elliptical Slice Sampling: An MCMC algorithm for sampling continuous probability distributions in parallel

Sample usage is shown in the script example.py, which can be run on an EC2 cluster with sufficiently many cores.

Gess is built in Python, and it uses the IPython.parallel module to scale MCMC over large clusters. Depending on the version of IPython that you run, you may find that it tries to cache everything, which will cause your experiment to crash. The various caches can be cleared as follows:

    def clear_cache(rc, dview):
        rc.purge_results('all') #clears controller
        rc.results.clear()
        rc.metadata.clear()
        dview.results.clear()
        assert not rc.outstanding, "don't clear history when tasks are outstanding"
        rc.history = []
        dview.history = []

Alternatively, it may suffice to use this command:

    %reset
