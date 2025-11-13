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

            other = other if isinstance(other, Value) else Value(other)
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
            out = Value(self.data + other.data, (self, other), '+')

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

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                         build_topo(child)
                    topo.append(v)

            build_topo(self) 


            for v in reversed(topo):
                v._backward()