import math 

# given info
x = 1.0 # input if x = 0.0 then no update shown
y = 0.0 # target result 

# model info
w = 0.5 # initial weight
b = 0.0 # initial bias
lr = 0.1 # learning rate
print(f'weight: {w}')

# model
z = w*x + b 
a = 1/(1+ math.exp(-z))
loss = (a-y)**2

#gradient from note, it comes from mathematical calculation
dl_da = 2*(a-y) 
da_dz = a*(1-a)
dz_dw = x

dl_dw = dl_da*da_dz*dz_dw

# new weight , gradient descent = lr*dl_dw
w -= lr*dl_dw

print(f'new weight: {w}')


