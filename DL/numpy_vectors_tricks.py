import numpy as np

# Doesn't work consistently, DON'T USE
a = np.random.rand(5)

print(a)
# [0.58948977 0.64802942 0.51805619 0.50749113 0.01336859]

print(a.shape)
# (5,) (rank 1) <--- bad practice

print(a.T)
# [0.58948977 0.64802942 0.51805619 0.50749113 0.01336859]

print(np.dot(a, a.T))
# 1.7393628094789653

# Instead, do this:
a = np.random.randn(5, 1)
print(a)

# [[-0.24049363]
#  [-0.95443891]
#  [-1.39525361]
#  [ 0.53591628]
#  [ 0.92730926]]

print(a.T)
# [[ 0.63081868 -0.4068449   1.27419215 -0.59172414  0.16843428]]

print(np.dot(a, a.T))
#[[ 0.43310391  0.46172977  0.1357457   0.58961507 -1.02745102]
# [ 0.46172977  0.49224764  0.14471777  0.62858547 -1.09536004]
# [ 0.1357457   0.14471777  0.04254613  0.18480025 -0.3220291 ]
# [ 0.58961507  0.62858547  0.18480025  0.8026848  -1.39874192]
# [-1.02745102 -1.09536004 -0.3220291  -1.39874192  2.43741873]]

b = np.random.randn(8, 1)
print(b)
print(b.reshape(2, 2, 2))

a = np.array([[2, 1],
              [1, 3]])
print(a*a)


a=np.random.randn(3,3)
b=np.random.randn(3,1)
print(a*b)

a=np.array([[2,1],[1,3]])
print(np.dot(a,a))

a=np.array([[1,1],[1,-1]])
b=np.array([[2],[3]])
print(a+b)

a = np.random.randn(1, 3)
b = np.random.randn(3, 3)
print((a*b).shape)

print(np.mean(a))

x = np.random.randn(4, 5)
y = np.sum(x, axis=1)
print(y.shape)