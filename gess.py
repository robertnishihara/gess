import numpy
from IPython import parallel

import fit_mvstud

def ess_update(x0, mu, cholSigma, logl, logf, tparams, cur_log_lklhd=None):
    calls = 0
    x0 = x0 - mu
    dim = x0.size
    assert cholSigma.shape == (dim,dim), "cholSigma is wrong size"
    
    if cur_log_lklhd is None:
        cur_log_lklhd = logl(x0 + mu, logf, tparams)
        calls += 1

    # Set up the ellipse and the slice threshold
    nu = numpy.dot(cholSigma, numpy.random.normal(size=dim))
    u = numpy.log(numpy.random.random())
    h = u + cur_log_lklhd

    # Bracket whole ellipse with both edges at first proposed point
    phi = numpy.random.random()*2*numpy.pi
    phi_min = phi - 2*numpy.pi
    phi_max = phi

    # Slice sample the loop
    while True:
        # Compute x_prop for proposed angle difference and check if it
        # is on the slice
        x_prop = x0*numpy.cos(phi) + nu*numpy.sin(phi)
        cur_log_lklhd = logl(x_prop + mu, logf, tparams)
        calls += 1
        if cur_log_lklhd > h:
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise Exception("Shrunk to current position and still not acceptable.")
        # Propose new angle difference
        phi = numpy.random.random()*(phi_max - phi_min) + phi_min

    x1 = x_prop + mu
    return x1, calls, cur_log_lklhd, nu

def tess_update(x0, mu, Sigma, invSigma, cholSigma, nu, logl, logf, thinning, repeats):
    dim = x0.size

    tparams = (dim, mu, invSigma, nu)

    x1 = x0
    cur_log_lklhd = logl(x1, logf, tparams)
    xs = numpy.zeros((repeats / thinning,dim))
    calls = 1

    for i in xrange(repeats):
        # Sample p(s|x)
        alpha = (dim + nu) / 2
        beta = (nu + numpy.dot(x1-mu,numpy.dot(invSigma,x1-mu))) / 2
        s = 1./numpy.random.gamma(alpha, 1./beta)

        # Sample p(x|s)
        chol_Sigma = numpy.sqrt(s)*cholSigma # this equals chol(s*Sigma)
        x1, new_calls, cur_log_lklhd, ess_nu = ess_update(x1, mu, chol_Sigma, logl, logf, tparams, cur_log_lklhd)
            
        if (i + 1) % thinning == 0:
            xs[(i + 1) / thinning - 1] = x1
        calls += new_calls
    return xs, calls

def parallel_tess_update(group1, group2, logf, thinning, repeats, dview=None):
    (n,dim) = group1.shape
    t_mu, t_Sigma, t_nu = fit_mvstud.fit_mvstud(group2)
    if t_nu == numpy.Inf:
        t_nu = 1e6
    t_inv_Sigma = numpy.linalg.inv(t_Sigma)
    try:
        t_chol_Sigma = numpy.linalg.cholesky(t_Sigma)
    except:
        import cPickle
        cPickle.dump(t_Sigma, open('output/t_sigma_fail.pickle', 'wb'))

    def logl(x, logf, tparams):
        # tparams is (dim, mu, invSigma, nu)
        def logt(x, tparams):
            (dim, mu, invSigma, nu) = tparams
            return -(dim+nu)/2*numpy.log(1+numpy.dot(x-mu,numpy.dot(invSigma,x-mu))/nu)
        return logf(x) - logt(x, tparams)

    if dview is None:
        print "Note that parallelism is not being used."
        results = map(tess_update, group1, n*[t_mu], n*[t_Sigma], n*[t_inv_Sigma], n*[t_chol_Sigma], n*[t_nu], n*[logl], n*[logf], n*[thinning], n*[repeats])
    else:
        results = dview.map_sync(tess_update, group1, n*[t_mu], n*[t_Sigma], n*[t_inv_Sigma], n*[t_chol_Sigma], n*[t_nu], n*[logl], n*[logf], n*[thinning], n*[repeats])

    samples = numpy.zeros((n,repeats / thinning,dim))
    calls = 0
    for i in xrange(n):
        samples[i,:,:] = results[i][0]
        calls += results[i][1]

    return samples, calls

def parallel_gess(chains, iters, burnin, thinning, starts, logf, repeats, dview=None):
    dim = starts.shape[1]
    assert starts.shape[0] == 2*chains, "starts is wrong shape"
    assert iters % repeats == 0, "iters must be divisible by repeats"
    assert burnin % repeats == 0, "burnin must be divisible by repeats"
    assert iters % thinning == 0, "iters must be divisible by thinning"
    assert burnin % thinning == 0, "burnin must be divisible by thinning"

    group1 = starts[:chains,:]
    group2 = starts[chains:,:]
    samples = numpy.zeros((2 * chains, iters / thinning, dim))
    calls = 0

    # do the sampling
    for i in xrange(-burnin / repeats, iters / repeats):
        samples1, calls1 = parallel_tess_update(group1, group2, logf, thinning, repeats, dview)
        group1 = samples1[:,repeats / thinning -1,:]
        samples2, calls2 = parallel_tess_update(group2, group1, logf, thinning, repeats, dview)
        group2 = samples2[:,repeats / thinning -1,:]

        if i >= 0:
            # update samples
            for j in xrange(chains):
                samples[j,i*repeats/thinning:(i+1)*repeats/thinning,:] = samples1[j]
                samples[chains+j,i*repeats/thinning:(i+1)*repeats/thinning,:] = samples2[j]
            calls += (calls1 + calls2)

    return samples, calls
