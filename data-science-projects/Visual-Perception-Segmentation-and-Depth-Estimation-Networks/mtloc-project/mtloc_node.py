
#Neural ODES mit ASM
from scipy.integrate import ode
import torch
import torchvision
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

# to do: create ODE dynamics of the net for forward and backward pass for layer creation



#def dopri5_solver(z0, t_range_forward, dynamics):

#    solver = ode(dynamics)
#    solver.set_integrator('dopri5')

#    m = 1
#    solver.set_f_params(m)

#    delta_t = t_range_forward[1] - t_range_forward[0]
#    zt = z0.clone()

#    t0 = t_range_forward[0]
#    solver.set_initial_value(zt, t0)
#    t1 = t_range_forward[0]
#    N = 75
#    t = np.linspace(t0, t1, N)
#    sol = np.empty((N, 4))
#    sol[0] = zt

#    # Repeatedly call the `integrate` method to advance the
#    # solution to time t[k], and save the solution in sol[k].
#    k = 1
#    while solver.successful() and solver.t < t1:
#        solver.integrate(t[k])
#        sol[k] = solver.y
#        k += 1
#        solver.integrate(t_range_forward[0])
#        #for tf in range(t_range_forward):
#        #z = z + h * f(z, t)
#        #t = t + h
#    return sol[k]



def euler_solver(z0, t_range_forward, dynamics):
    """
    Simplest Euler ODE initial value solver
    """
    delta_t = t_range_forward[1] - t_range_forward[0]
    zt = z0.clone()

    #make forward euler method
    for tf in t_range_forward:
        f = dynamics(zt, tf)
        #update
        zt = zt + delta_t*f
    return zt


class ODELayerFunc(torch.autograd.Function):

    @staticmethod
    def forward(context, z0, t_range_forward, dynamics, *theta):
        #initialize step size
        delta_t = t_range_forward[1] - t_range_forward[0]

        zt = euler_solver(z0, t_range_forward, dynamics)
        #zt = dopri5_solver(z0, t_range_forward, dynamics)

        context.save_for_backward(zt, t_range_forward, delta_t, *theta)
        context.dynamics = dynamics

        #return final evaluation of the state zt
        return zt


    def backward(context, adj_end):
        #unpack values saved in forward pass
        zT, t_range_forward, delta_t, *theta = context.saved_tensors
        dynamics = context.dynamics
        t_range_backward = torch.flip(t_range_forward, [0,])

        zt = zT.clone().requires_grad_()
        adjoint = adj_end.clone()
        #an accumulator for the parameter gradients
        dLdp = [torch.zeros_like(p) for p in theta]

        for tb in t_range_backward:
            #create graph and compute all the vector-jacobian products
            with torch.set_grad_enabled(True):
                f = dynamics(zt, tb)
                adjoint_dynamics, *dldp_ = torch.autograd.grad([-f], [zt, *theta], grad_outputs =[adjoint])


            for i, p in enumerate(dldp_):
                #update param grads
                dLdp[i] = dLdp[i] - delta_t * p

            #update the adjoint
            adjoint = adjoint - delta_t*adjoint_dynamics
            #backward in time euler
            zt.data = zt.data - delta_t * f.data

        return (adjoint, None, None, *dLdp)




class ODELayer(torch.nn.Module):


    def __init__(self, dynamics, t_start=0., t_end=1., granularity=25):
        super().__init__()

        self.dynamics = dynamics
        self.t_start, self.t_end, self.granularity = t_start, t_end, granularity
        self.t_range = torch.linspace(self.t_start, self.t_end, self.granularity)


    def forward(self, input):
        return ODELayerFunc.apply(input, self.t_range, self.dynamics, *self.dynamics.parameters())



class Dynamics(torch.nn.Module):
    #Define the dynmaics of the ODE
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        self.linear = torch.nn.Linear(self.n_dim, self.n_dim)

    def forward(self, z, t):
        #print('state shape: ', z.shape)
        #print('timepoint shape: ', t.shape)

        return torch.tanh(self.linear(z))*t

class ODEClassifier(torch.nn.Module):
    #Model: ODElayer + a classifier

    def __init__(self, n_dim, n_classes, ode_dynamics, t_start = 0., t_end = 1., granularity = 50):
        super().__init__()
        self.n_dim, self.n_classes = n_dim, n_classes
        self.t_start, self.t_end, self.granularity = t_start, t_end, granularity

        self.odelayer = ODELayer(ode_dynamics, self.t_start, self.t_end, self.granularity)
        self.classifier = torch.nn.Linear(self.n_dim, self.n_classes)


    def forward(self, x, drop_ode=False):
        if not drop_ode:
            x = self.odelayer(x)
        return F.log_softmax(self.classifier(x), 1)

if __name__ == '__main__':
   
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, required=False, default=1e-3, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=3, help='Total epochs')
    parser.add_argument('--drop_ode', action='store_true', help='Just drop the ODE Layer (for comparison)')
    args = parser.parse_args()

    dynamics = Dynamics(28 * 28)
    print('dynamics: ', dynamics)
    model = ODEClassifier(28 * 28, 10, dynamics)
    print('model: ', model)
    
    #summary(model, (1, 28, 28))

    if torch.cuda.is_available():
        model = model.cuda()
    
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    mnist = torchvision.datasets.MNIST('./mnist', download=True, train=True, transform=torchvision.transforms.ToTensor())
    mnisttest = torchvision.datasets.MNIST('./mnist', download=True, train=False, transform=torchvision.transforms.ToTensor())
    mnistdl = torch.utils.data.DataLoader(mnist, shuffle=True, batch_size=args.batch_size, drop_last=True, pin_memory=True)
    mnisttestdl = torch.utils.data.DataLoader(mnisttest, shuffle=True, batch_size=args.batch_size, drop_last=True, pin_memory=True)

    for e in range(args.epochs):
        for i, (X, Y) in enumerate(mnistdl):
            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()
            X = X.view(args.batch_size, -1) # flatten everything
            #print(X.shape)

            output = model(X, drop_ode=args.drop_ode)
            loss = F.nll_loss(output, Y)
            
            if i % 20 == 0:
                print(f'[Training] {i}/{e}/{args.epochs} -> Loss: {loss.item()}')

            optim.zero_grad()
            loss.backward()
            optim.step()

        total, correct = 0, 0
        for j, (X, Y) in enumerate(mnisttestdl):
            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()
            X = X.view(args.batch_size, -1) # flatten everything
            
            output = model(X, drop_ode=args.drop_ode)
            correct += (output.argmax(1) == Y).sum().item()
            total += args.batch_size

        accuracy = (correct / float(total)) * 100
        print(f'[Testing] -/{e}/{args.epochs} -> Accuracy: {accuracy}')
                































