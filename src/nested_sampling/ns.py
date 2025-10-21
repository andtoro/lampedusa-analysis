import numpy as np
import os
"""
to install cpnest, make sure to clone the repository from github:
git clone git@github.com:johnveitch/cpnest.git
cd cpnest
git checkout massively_parallel
git pull
python setup.py install
"""
import raynest.model

def poly(x, p, order = 1):
    p = np.sum(np.array([p['{}'.format(i)]*x**i for i in range(order)]))
    return p

class PolynominalModel(raynest.model.Model):
    """
    An n-dimensional gaussian
    """
    def __init__(self, data, order=1):
        self.data_x  = data[:,0]
        self.data_y  = data[:,1]
        self.sigma_x = data[:,2]
        self.sigma_y = data[:,3]
        self.order   = order+1
        self.names=['{0}'.format(i) for i in range(self.order)]
        self.bounds=[[-10,10] for _ in range(self.order)]
        # add the unobserved x data points
        for i in range(self.data_x.shape[0]):
            self.names.append('x_{}'.format(i))
            self.bounds.append([self.data_x[i]-5*self.sigma_x[i],self.data_x[i]+5*self.sigma_x[i]])

    def log_likelihood(self,p):
        model = np.array([poly( p['x_{}'.format(i)], p, order=self.order) for i in range(self.data_x.shape[0])])
        logL_y = -0.5*np.sum(((self.data_y-model)/self.sigma_y)**2)
        logL_x = 0.0
        for i in range(self.data_x.shape[0]):
            logL_x += -0.5*((self.data_x[i]-p['x_{}'.format(i)])/self.sigma_x[i])**2
        return logL_x+logL_y
    
    def log_prior(self,p):
        logP = super(PolynominalModel,self).log_prior(p)
        return logP

if __name__=='__main__':
    # hard coded options
    out_folder = 'linear_quick'
    order      = 1
    
    data = np.loadtxt('data_obs.txt', usecols=(0,1,3,4))
    data = data[np.argsort(data[:,0]),:]
#    import matplotlib.pyplot as plt
#    plt.errorbar(data[:,0],data[:,1],xerr=data[:,2],yerr=data[:,3])
#    plt.show()
#    print(data)
#    exit()
    M=PolynominalModel(data, order = order)

    if 1:
        work=raynest.raynest(M, verbose=2,
                           nnest=1, nensemble=3, nlive=100, maxmcmc=20, nslice=0, nhamiltonian=0, seed = 1,
                           resume=1, periodic_checkpoint_interval=600, output=out_folder)
        work.run()
        print("estimated logZ = {0} \pm {1}".format(work.logZ,work.logZ_error))
    
        samples = work.posterior_samples
    else:
        import h5py
        filename = os.path.join(out_folder,"raynest.h5")
        h5_file = h5py.File(filename,'r')
        samples = h5_file['combined'].get('posterior_samples')

    
    models = []
    for s in samples:
        models.append(np.array([poly( s['x_{}'.format(i)], s, order=M.order) for i in range(M.data_x.shape[0])]))
    
    l,m,h = np.percentile(models, [5,50,95], axis=0)
    
    import matplotlib.pyplot as plt
    
    f = plt.figure()
    ax = f.add_subplot(211)
    ax.hist(samples['0'], bins=100, density=True)
    ax.set_xlabel('intercept')
    ax = f.add_subplot(212)
    ax.hist(samples['1'], bins=100, density=True)
    ax.set_xlabel('slope')
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(M.data_x, m,'-k')
#    for ms in models:
#        ax.plot(M.data_x, ms,'-k',linewidth=0.1)
    
    ax.errorbar(M.data_x,M.data_y,xerr=M.sigma_x,yerr=M.sigma_y,linestyle='')
    ax.fill_between(M.data_x,l,h,facecolor='turquoise')
    
    plt.show()