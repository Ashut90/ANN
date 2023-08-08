import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import joblib 
import os 
from matplotlib.colors import ListedColormap



class Perceptron:
    def __init__(self , lr , epochs): # initiate your weight here 
        self.weights = np.random.randn(3) * 1e-4 # initiating the smallest random weight 
        print(f"Weight at initial stange: {self.weights}")
        self.lr = lr
        self.epochs = epochs

    def activationfunction(self, input , weights): # activtion functiin = i/p*weight
        z = np.dot(input , weights) # np.dot = used to find the vector matrix product
        return np.where(z > 0 , 1,0)

    def fit(self,X,y): # define the training method here , Its a complex part 
        self.X = X
        self.y = y
        X_bias = np.c_[self.X, -np.ones((len(self.X),1))] # Bias = influencing factor
        print(f"X_bias: \n{X_bias}")

        for epoch in range(self.epochs):
            print("--"*8)
            print(f"Epochs: {epoch}")
            print("--"*8)

            y_hat = self.activationfunction(X_bias , self.weights) # Forward pass
            print(f"predicted value after forward pass: \n{y_hat}")
            self.error = self.y - y_hat
            print(f"error: \n{self.error}")
            self.weights = self.weights + self.lr * np.dot(X_bias.T , self.error) # backward propagation
            print(f"Updated weight after epoch: \n{epoch}/{self.epochs} : \n{self.weights}") 
            print("**"*10)

    def predict(self,X):
        X_bias = np.c_[X ,-np.ones((len(X),1))]
        return self.activationfunction(X_bias , self.weights)


    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"total_loss: \n{total_loss}")
        return total_loss
    
def prep_data(df):
    X = df.drop("y" , axis = 1)
    y = df["y"]

    return X ,y
    
XOR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y":  [0,1,1,0]
    }
    
df = pd.DataFrame(XOR)

X,y = prep_data(df)
lr = 0.3
epoch = 10

model_xor = Perceptron(lr = lr , epochs = epoch)
model_xor.fit(X,y)
loss = model_xor.total_loss()

def model_save(model_xor , filename):
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  filePath = os.path.join(model_dir, filename) # model/filename
  joblib.dump(model_xor, filePath)

model_save(model_xor , "xor.model")


#plot the data 

def save_plot(df, file_name, model):
  def _create_base_plot(df):
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))]) # type: ignore

    X = X.values # as a array
    x1 = X[:, 0] 
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1 , x1.max() + 1
    x2_min, x2_max = x2.min() -1 , x2.max() + 1  

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    print(xx1)
    print(xx1.ravel())
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.plot()



  X, y = prep_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)

save_plot(df , "xor.png" , model_xor)


 