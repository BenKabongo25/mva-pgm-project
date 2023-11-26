import numpy as np


class MAGMA:
    def __init__(self,times,n_indiv,m_init=0,theta_init=None,sigma_init=1):
        """
        Iniitialize values

        We are making some assumption in the first version of this code :
        - all individuals have the same set of time stamp
        - we do not use repeated short runs
        """
        self.n_indiv = n_indiv # Number of individuals 
        self.n_times = len(times) #Number of time stamps

        self.m0 = np.array([m_init for i in range(self.n_times)]) #initialize m0 with zeros

        if theta_init==None: # we initilize theta with random values
            self.theta0 = (1+0.05*np.random.random(),1+0.05*np.random.random())
            self.theta = np.array([(1+0.05*np.random.random(),1+0.05*np.random.random()) for i in range(n_indiv)])

        
        else:
            self.theta0 = theta_init[0]
            self.theta = theta_init[1:]

        self.sigma = np.array([sigma_init for i in range(n_indiv)]) #initialization of sigma
        self.time = times # array of timestamps

    def compute_kernels(self,theta0,theta,sigma):
        """
        Compute kernels we need in our algorithm
        """

        self.kernel_Ktheta = self.kernel_EQ(theta0,self.time.reshape(-1,1),self.time.reshape(1,-1))
        self.kernel_c = [self.kernel_EQ(th,self.time.reshape(-1,1),self.time.reshape(1,-1)) for th in theta]
        self.kernel_psi = np.array([self.kernel_c[i]+sigma[i]*np.identity(self.n_times) for i in range(self.n_indiv)])


    def gradient_descent(self,f_derivative,x0,eps=1e-3,gamma=0.05):
        """
        gradient descent to find the maximum of f

        The derivative of f is f_derivative

        We start our search from x0
        
        """
        delta = np.linalg.norm(f_derivative(x0))
        x = x0
        previous_x = x0

        while delta>eps:
            previous_x=x
            x = previous_x+gamma*f_derivative(previous_x)
            delta = np.sum(f_derivative(x)**2)
            print(delta)

        return x
            

    def kernel_EQ(self,theta,x1,x2):
        """
        Compute the exponential quadratic kernel for parameters theta and values x1 and x2
        """
        return theta[0]**2*np.exp(-(x1-x2)**2/(2*theta[1]**2))
    
    def derivative_lognormal(self,x,m,S):
        """
        The derivative of log(N(x,m,S))
        m is the mean
        S is the covariance function
        """
        inv_S = np.linalg.inv(S)
        return -1/2*inv_S.T+1/2*inv_S.T@(x-m).reshape(-1,1)@(x-m).reshape(1,-1)@inv_S.T

    def gradient_likelyhood_theta0(self,theta):
        """
        The gradient of the function we have to maximize in the M step to get theta0
        """

        Ktheta = self.kernel_EQ(theta,self.time.reshape(-1,1),self.time.reshape(1,-1))
        inv_Ktheta = np.linalg.inv(Ktheta)
        square_time = (self.time.reshape(-1,1)-self.time.reshape(1,-1))**2 # Matrix (t_i t_j)_(i,j)
        derivative_likelihood = self.derivative_lognormal(self.m0,self.m0_estim,self.kernel_Ktheta)+ 1/2*inv_Ktheta@self.kernel_K.T@inv_Ktheta
        derivative0 = np.sum(derivative_likelihood*2*Ktheta/theta[0])# Apply the chain rule
        derivative1 = np.sum(derivative_likelihood*Ktheta*square_time/theta[1]**3) # Apply the chain rule
        return np.array([derivative0, derivative1])


    def gradient_likelyhood_thetasigma_i(self,thetasigma):
        """
        The gradient of the function we have to maximize in the M step to get theta and sigma
        """
        theta_i,sigma_i=thetasigma[0:2],thetasigma[2]
        kernel_c = self.kernel_EQ(theta_i,self.time.reshape(-1,1),self.time.reshape(1,-1))
        kernel_psi = self.kernel_c+sigma_i*np.identity(self.n_times)
        inv_kernel_psi = np.linalg.inv(kernel_psi)
        square_time = (self.time.reshape(-1,1)-self.time.reshape(1,-1))**2

        derivative_likelihood = self.derivative_lognormal(self.ylocal,self.m0_estim,self.kernel_psi)+ 1/2@inv_kernel_psi@self.kernel_K.T@inv_kernel_psi

        derivative0 = np.sum(derivative_likelihood*2*kernel_c/theta_i[0])# Apply the rule chain
        derivative1 = np.sum(derivative_likelihood*kernel_c*square_time/theta_i[1]**3)# Apply the rule chain
        derivative_sigma = np.sum(derivative_likelihood*2*sigma_i*np.identity(self.n_times))# Apply the rule chain
        return np.array([derivative0,derivative1,derivative_sigma])

    def E_step(self,y):
        # compute the estimator kernel K
        self.kernel_K = np.linalg.inv(np.linalg.inv(self.kernel_Ktheta)+np.sum([np.linalg.inv(Psi) for Psi in self.kernel_psi]))

        # compute the estimator m0
        compute_psi = np.linalg.inv(self.kernel_Ktheta)@self.m0 + np.sum([np.linalg.inv(self.kernel_psi[i])@y[i] for i in range(self.n_indiv)])
        self.m0_estim = self.kernel_K@compute_psi

    def M_step(self,y):
        """
        To get the optimal parameters, we need to maximize likelihood with gradient descent
        """

        self.theta0 =  self.gradient_descent(self.gradient_likelyhood_theta0,self.theta0)
        for i in range(self.n_indiv):
            self.ylocal=y[i] #We need to know y[i] to compute gradient_likelyhood_thetasigma_i
            result = self.gradient_descent(self.gradient_likelyhood_thetasigma_i,np.array([self.theta[i,0],self.theta[i,1],self.sigma[i]]))
            self.theta = result[0:2]
            self.sigma = result[2]
            self.compute_kernels()

    def fit(self,y,eps):
        """
        We need to change the convergence criteria
        the criterion depends on the likelihood
        """
        self.compute_kernels(self.theta0,self.theta,self.sigma)
        criterion = 1
        while criterion > eps: # we have to change the criterion
            self.E_step(y)
            self.M_step(y)
            criterion=criterion/2



model = MAGMA(np.linspace(0,10,11),20)
y = 2*np.random.random((20,11))+1
model.fit(y,0.1)



