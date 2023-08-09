from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
from utils.all_utils import save_model, save_plot


NAND = {
    "x1": [0,1,0,1],
    "x2": [0,0,1,1],
    "y": [1,1,1,0] 
}

df = pd.DataFrame(NAND)

X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()
save_model(model, "nand.model")
save_plot(df , "nand.png" , model)

