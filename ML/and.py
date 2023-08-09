from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
from utils.all_utils import save_model, save_plot


AND = {
    "x1": [0,1,0,1],
    "x2": [0,0,1,1],
    "y": [0,0,0,1] 
}

df = pd.DataFrame(AND)

X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()
save_model(model, "and.model")
save_plot(df , "and.png" , model)

