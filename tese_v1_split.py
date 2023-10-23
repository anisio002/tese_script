import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import csv
import time
from math import sqrt
from TeseFunctios import TeseFunctios as tese

save_path = "/home/anisio/Documentos/Doutorado/Tese/Dados/tese/models/"
path = "/home/anisio/Documentos/Doutorado/Tese/Dados/Pareados"
file_path = "/home/anisio/Documentos/Doutorado/Tese/Dados/tese/rede/"
#Dataset utilizado para validação
dadosParaiso = pd.read_csv(path + '/GO-ALTO PARAISO DE GOIAS.csv',sep=',')

dadosCampinaVerde = pd.read_csv(path + '/MG-CAMPINA VERDE.csv',sep=',')
dadosSorriso = pd.read_csv(path + '/MT-SORRISO.csv',sep=',')
dadosDiamante = pd.read_csv(path + '/PR-DIAMANTE DO NORTE.csv',sep=',')
dadosCampo = pd.read_csv(path + '/RS-CAMPOBOM.csv',sep=',')
dadosBarcelos = pd.read_csv(path + '/AM-BARCELOS.csv',sep=',')
dadosMacapa = pd.read_csv(path + '/AP-MACAPÁ.csv',sep=',')
dadosPetrolina = pd.read_csv(path + '/PE-PETROLINA.csv',sep=',')
dadosIrece = pd.read_csv(path + '/BA-IRÊCE.csv',sep=',')

listNames = ["Campo Bom - RS","Sorriso - MT","Diamante do Norte - PR","Campina Verde - MG","Barcelos - AM", "Macapá - AP", "Petrolina - PE", "Irêce - BA"]
listaDados = [dadosCampo,dadosSorriso,dadosDiamante,dadosCampinaVerde,dadosBarcelos,dadosMacapa,dadosPetrolina,dadosIrece]
data = pd.read_csv(path + '/AP-MACAPÁ.csv',sep=',')

fields = ['City', 'TrainningSize','RMSE','Time', 'Epoch','Training_Loss', 'Training_Accuracy', 'Test_Loss', 'Test_Accuracy', 'RMSE_UNKNOW']
functions = ['f1','rmse']
training_sets = [0.2,0.4,0.6]

pos = ['umid_inst', 'pto_orvalho_inst', 'pressao', 'radiacao', 'vento_vel']

y_validation = dadosParaiso['temp_inst'].to_numpy()
X_validation= dadosParaiso.drop(['temp_inst'],axis = 1)
scaler = StandardScaler().fit(X_validation)
X_validation = scaler.transform(X_validation)

epochs = 1
seed = 7
learning_rate = 1e-4
min_delta = 1e-4
patience = 1

pinn = []
dl = []
rmse = []
rmse_dl = []
mlp_time=[]

class HeatEquationLoss(tf.keras.losses.Loss):
    def __init__(self, name='heat_equation_loss', **kwargs):
        super(HeatEquationLoss, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return tese.heat_equation_loss(y_true, y_pred)

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('.', end=' ')
        #if (epoch % 10) == 0:
        print(f"Epoch {epoch+1}: loss = {logs['loss']} rmse= {logs['root_mean_squared_error']}")
callback = MyCallback()
early_stopping = EarlyStopping(monitor= 'root_mean_squared_error', min_delta=min_delta, patience=patience)

    
contCity =0
for data in listaDados:
    city = listNames[contCity]
    
    y = data['temp_inst'].to_numpy()
    X = data.drop(['temp_inst'],axis = 1)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    
    pipelines = []
    pipelines.append(('LR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
    pipelines.append(('LASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
    pipelines.append(('EN', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
    pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
    pipelines.append(('CART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))
    pipelines.append(('SVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
    pipelines.append(('PINN', Pipeline([('Scaler', StandardScaler()),('PINN', None)])))
    pipelines.append(('MLP', Pipeline([('Scaler', StandardScaler()),('MLP', None)])))
    
    for name, model in pipelines:
        file_name = file_path + 'Split_' + name + '_' + city
        with open(file_name + '.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            
    for test_size in training_sets:
    
        for name, model in pipelines:
            results=[]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
                
            history= None
            num_epochs = None
            y_pred = None
            train_score = None
            test_score = None
            y_unknow = None
            
            inicio = time.time()
            if name not in ["MLP", "PINN"]:
                print(">>>>"+name)
                model.fit(X_train, y_train)
                train_score = [None, model.score(X_train, y_train)] 
                test_score = [None, model.score(X_test, y_test),None ] 
                y_pred = model.predict(X_test)
                y_unknow = model.predict(X_validation)
            else:
                model_rn = None
                if name == "MLP":
                    print(">>>>"+name)
                    model_rn = tese.build_model(X_train.shape[1])
                    model_rn.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[RootMeanSquaredError()], optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
                if name == "PINN":
                    print(">>>>"+name)
                    model_rn = tese.build_model(X_train.shape[1])
                    model_rn.compile(loss=HeatEquationLoss(), metrics=[RootMeanSquaredError()], optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
                history = model_rn.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2, verbose=0, callbacks=[callback, early_stopping])
                train_score = model_rn.evaluate(X_train, y_train)  # Avalie o modelo
                test_score = model_rn.evaluate(X_test, y_test)
                y_pred = model_rn.predict(X_test)
                y_unknow = model_rn.predict(X_validation)
                num_epochs = len(history.epoch)
                tese.plot_rmse(history, None,file_name,city,name,test_size)
            fim = time.time()
                
            mlp_time = fim - inicio
            print(f'Testsize: {str(test_size)} - Localidade: {city}')
            results.append([city, str(test_size), sqrt(mean_squared_error(y_test, y_pred)), mlp_time, num_epochs, train_score[0],train_score[1],test_score[0],test_score[1],sqrt(mean_squared_error(y_validation, y_unknow))])
            values = ','.join(str(v) for v in results)
            with open(file_name + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(results)
                print("List saved to CSV file successfully.")
    contCity += 1            
