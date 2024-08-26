import numpy as np
import scipy.stats as stats
class DoubleCosine(stats.rv_continuous):
    def __init__(self, c=np.pi, a=0, b=2*np.pi):
        super().__init__()
        self.a = a
        self.b = b
        self.scale = (b-a)/(2*np.pi)
        self.loc = np.pi / self.scale + a
        self.start = a
        self.end = 2*np.pi*self.scale
        self.c = c
        self.scale1 = (self.c - self.start)/np.pi
        self.scale2 = (self.end - self.c)/np.pi
        self.dist1 = stats.cosine(loc=c+self.start, scale=self.scale1)
        self.dist2 = stats.cosine(loc=c+self.start, scale=self.scale2)
    
    def _pdf(self, x):

        x = np.asarray(x)

        c = self.c

        out = np.empty_like(x)
        out[x <= c] = self.dist1.pdf(x[x <= c]) * self.scale1 * 2
        out[x > c] = self.dist2.pdf(x[x > c]) * self.scale2 * 2

        return out
    
    def _cdf(self, x):
        x = np.asarray(x)

        c = self.c

        out = np.empty_like(x)
        out[x <= c] = self.dist1.cdf(x[x <= c]) *  self.scale1 * 2
        Fc1 = self.dist1.cdf(c)
        Fc2 = self.dist2.cdf(c)
        out[x > c] = (self.dist2.cdf(x[x > c]) - Fc2) * self.scale2 * 2 + Fc1 * self.scale1 * 2

        return out

    def  _stats(self):
        m = ((np.pi**2 - 4)* self.scale * 2 * np.pi + 8*self.c) / (2*np.pi**2)
        v = None
        s = None
        k = None
        return m, v, s, k
    
    def _ppf(self, q):

        q = np.asarray(q)

        c = self.c

        out = np.empty_like(q)
        Fc1 = self.dist1.cdf(c)
        Fc2 = self.dist2.cdf(c)
        out[q <= Fc1 * self.scale1 * 2] = self.dist1.ppf(q[q <= Fc1 * self.scale1 * 2] / self.scale1 / 2)
        out[q > Fc1 * self.scale1 * 2] = self.dist2.ppf((q[q > Fc1 * self.scale1 * 2] - Fc1 * self.scale1 * 2) / self.scale2 / 2 + Fc2) 

        return out
    
    def _rvs(self, size, *args, **kwargs):

        u = np.random.uniform(0, 1, size)
        Fc1 = self.dist1.cdf(self.c)
        Fc2 = self.dist2.cdf(self.c)
        Fc = Fc1 * self.scale1 * 2

        x = np.empty(size)
        x[u <= Fc] = self.dist1.ppf(u[u <= Fc] / self.scale1 / 2)
        x[u > Fc] = self.dist2.ppf((u[u > Fc] - Fc) / self.scale2 / 2 + Fc2) 

        return x

class InvDist:
        def __init__(self, dist_type, **kwargs):
            self.dist = dist_type(**kwargs)
        def pdf(self, x, *args, **kwargs):
            return self.dist.pdf(-x, *args, **kwargs)
        def rvs(self, *args, **kwargs):
            return -self.dist.rvs(*args, **kwargs)
        
        # forward all other methods to the underlying dist
        def __getattr__(self, name):
            return getattr(self.dist, name)
