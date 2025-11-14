import math
from micrograd import Value, MLP, cross_entropy, eval_split
from utils import RNG, gen_data

random = RNG(42)

train_split, val_split, test_split = gen_data(random, n=100)

model = MLP(2, [16, 3])

learning_rate = 5e-3
beta1 = 0.9  
beta2 = 0.95  
weight_decay = 0


for p in model.parameters():
    p.m = 0.0 
    p.v = 0.0  


for step in range(100):

    if step % 10 == 0:
        val_loss = eval_split(model, val_split)  
        print(f"step {step}, val loss {val_loss:.6f}")
    
 
    loss = Value(0)  
    for x, y in train_split:
        logits = model([Value(x[0]), Value(x[1])])  
        loss += cross_entropy(logits, y)  
    loss = loss * (1.0 / len(train_split))  
    

    loss.backward()
    

    for p in model.parameters():
        p.m = beta1 * p.m + (1 - beta1) * p.grad 
        p.v = beta2 * p.v + (1 - beta2) * p.grad**2  
        m_hat = p.m / (1 - beta1**(step + 1)) 
        v_hat = p.v / (1 - beta2**(step + 1)) 
        p.data -= learning_rate * (m_hat / (v_hat**0.5 + 1e-8) + weight_decay * p.data) 
    
    model.zero_grad() 
    
    print(f"step {step}, train loss {loss.data}")