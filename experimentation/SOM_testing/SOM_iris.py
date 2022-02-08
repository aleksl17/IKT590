from matplotlib import transforms
from sklearn.utils import shuffle
from sklearn_som.som import SOM
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
#print(iris)
iris_data = iris.data
iris_label = iris.target

#iris_data = iris.data[:, :2]


som = SOM(m=3, n=1, dim=4)
som.fit(iris_data, epochs=100, shuffle=True)
transform = som.transform(iris_data)

print(transform)

#predictions = som.predict(iris_data)

# Plot the results
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,7))
x = iris_data[:,0]
y = iris_data[:,1]
colors = ['red', 'green', 'blue']

ax[0].scatter(x, y, c=iris_label, cmap=ListedColormap(colors))
ax[0].title.set_text('Actual Classes')
ax[1].scatter(transform[:,0], transform[:,1], c=iris_label, cmap=ListedColormap(colors))
ax[1].title.set_text('SOM Predictions')
plt.savefig(f'./.iris.png')