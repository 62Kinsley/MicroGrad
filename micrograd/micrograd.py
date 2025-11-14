import math
import random as std_random

class Value:
         # stores a single scalar value and its gradient 

        def __init__(self, data, _children=(), _op=''):

            self.data = data  # store value
            self.grad = 0   # store gradients

            self._backward = lambda: None   # Backpropagation function
            self._prev = set(_children)    #Predecessor node set
            self._op = _op     #Operation type, used for debugging and visualization


        def __add__(self, other): # (self:a  other:b)

            #out: a+b 
            #self.grad += out.grad
            #other.grad += out.grad

            #If other is not of type Value, then cast it to type Value.
            other = other if isinstance(other, Value) else Value(other) 

            # a new Value object with the sum of two values, predecessor nodes are self and other, operators .
            out = Value(self.data + other.data, (self, other), '+') 
            

            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward

            return out
        

        def __mul__(self, other): # (self:a  other:b)

            #out: a*b 
            #self.grad += other.data * out.grad 
            #other.grad += self.data * out.grad

            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), '*')

            def _backward():
                self.grad += other.data * out.grad 
                other.grad += self.data * out.grad
            out._backward = _backward

            return out
        


        def __pow__(self, other): # (self:a  other:b)

            #self.grad += (other * self.data**(other-1)) * out.grad

            assert isinstance(other, (int, float)), "only supporting int/float powers for now"
            out = Value(self.data**other, (self,), f'**{other}')

            def _backward():
                self.grad += (other * self.data**(other-1)) * out.grad
             
            out._backward = _backward

            return out
        

        def relu(self):
            out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

            def _backward():
                self.grad += (out.data > 0) * out.grad
            out._backward = _backward
            return out
        

        def tanh(self):
            out = Value(math.tanh(self.data), (self,), 'tanh')

            def _backward():
                self.grad +=  (1 - out.data**2) * out.grad
            out._backward = _backward

            return out


        def exp(self):
            out = Value(math.exp(self.data), (self,), 'exp')

            def _backward():
                self.grad +=  math.exp(self.data) * out.grad
                #self.grad +=  out.data * out.grad
            out._backward = _backward
            
            return out


        def log(self):
            out = Value(math.log(self.data), (self,), 'log')

            def _backward():
                self.grad += (1/self.data) * out.grad
            out._backward = _backward
            
            return out
        

        def __neg__(self): # -self

            return self * -1

        def __radd__(self, other):  # other + self

            return self + other

        def __sub__(self, other):  # self - other

            return self + (-other)
        
        def __rsub__(self, other):  # other - self

            return other + (-self)
        
        def __rmul__(self, other):  # other * self

            return self * other
        
        def __truediv__(self, other):  # self / other

            return self * other**-1
        
        def __rtruediv__(self, other):  # other / self

            return other * self**-1
        
        
        def backward(self):
            topo = []  #A list used to store the results of topological sorting.
            visited = set()

            #m = x * y
            #z = m + x
            #z._prev = (m, x)
            #m._prev = (x, y)
            #x._prev = ()
            #y._prev = ()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)#z, m, x
                    for child in v._prev:#v._prev = z._prev = (m, x)   v._prev = m._prev = (x, y)
                         build_topo(child) #m, x
                    topo.append(v) #x, y, m, z

            self.grad = 1.0 
            build_topo(self) 


            for v in reversed(topo):
                v._backward()



##The Module class itself does not directly participate in the computation of the neural network, but serves as the base class for other classes.
class Module: 
    def zero_grad(self): #Clear the parameter gradient to 0, pre train must do this
        for p in self.parameters(): 
            p.grad = 0

    def parameters(self):  # Return parameters， like w, b
        return [] 
    


class Neuron(Module):
    #Output = Activation function(w1*x1 + w2*x2 + ... + wn*xn + b)
    # Initialize the neuron, where nin is the number of inputs and nonlin indicates whether to use a non-linear activation function.
    def __init__(self, nin, nonlin=True):
        self.w = [Value(std_random.uniform(-1, 1) * nin**-0.5) for _ in range(nin)] #weight
        self.b = Value(0) #bias
        self.nonlin = nonlin 

    def __call__(self, x): # this method define how neuron input and 
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act #If the nonlin is True, the neuron uses the tanh activation function; otherwise, a linear function is used.
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'TanH' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs): # nin is input，nout is output
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class MLP(Module):

    def __init__(self, nin, nouts): # nin: number of inputs， nout: number of outputs
        sz = [nin] + nouts  #size of every layer
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))] #new a layer

    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) 
        return x
    
    def parameters(self):
         return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    


def eval_split(model, split):
    loss = Value(0) 

    for x, y in split:
        logits = model([Value(x[0]), Value(x[1])])

        loss += cross_entropy(logits, y)
    loss = loss * (1.0 / len(split))
    return loss.data 

def cross_entropy(logits, target):
    max_val = max(val.data for val in logits)# logits： [2.0, 1.0, 0.1]  max_val = 2.0
    logits = [val - max_val for val in logits] # [2-2, 1-2, 0.1-2] = [0, -1, -1.9]
    
    #logits are transformed into probability distributions.
    ex = [x.exp() for x in logits] #[exp(0), exp(-1), exp(-1.9)] ≈ [1, 0.3679, 0.1496]
    denom = sum(ex) #1 + 0.3679 + 0.1496 ≈ 1.5175
    probs = [x / denom for x in ex] #[1/1.5175, 0.3679/1.5175, 0.1496/1.5175] ≈ [0.659, 0.242, 0.099]

    logp = (probs[target]).log() #target = 0,  prob[target] = 0.659 ， logp = log(0.659) ≈ -0.417

    nll = -logp #-(-0.417) = 0.417  #Higher probability → smaller -log(prob) → smaller loss → more accurate
    return nll
