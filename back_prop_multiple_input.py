import math

# input

x = [1.0, 0.0, -1.0]
w = [0.5, -0.3, 0.8]
b = 0.1

y = 1.0
lr = 0.1

# compute z : z = w1.z1+.......+b

z = 0.0
for i in range(len(x)):
    z += w[i]*x[i]
z += b
print(f'initially z :{z}')
print(f'initially b :{b}')
print(f'initially w :{w}')



# activation and loss fxn
a = 1.0/(1+math.exp(-z))
loss = (a-y)**2

# gradient
dl_da = 2*(a-y)
da_dz = a*(1-a)

dl_dz = dl_da * da_dz

# dl_dz = dl_db ---> explain in note
dl_db = dl_dz

# dl_dw[i] = dl_da * x[i] ---> explain in note

dl_dw = [0.0]*len(w)
for i in range(len(w)):
    dl_dw[i] = dl_dz*x[i]

# gradient decent 
b -= lr*dl_db
for i in range(len(dl_dw)):
    w[i] -= lr*dl_dw[i]


z = 0.0
for i in range(len(x)):
    z += w[i]*x[i]
z += b
print(f'final z :{z}')
print(f'final b :{b}')
print(f'final w :{w}')



