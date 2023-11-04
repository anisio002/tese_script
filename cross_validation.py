from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import csv
import time
from math import sqrt
import sys
sys.path.insert(0, "/content/drive/MyDrive/Colab Notebooks")
from TeseFunctios import TeseFunctios as tese



save_path = "/content/drive/MyDrive/Doutorado/Dados/tese/PINN/"
path = "/content/drive/MyDrive/Doutorado/Dados/Pareados"
file_path = "/content/drive/MyDrive/Doutorado/Dados/tese/PINN/"

dadosParaiso = pd.read_csv(path + '/GO-ALTO PARAISO DE GOIAS.csv',sep=',')

dadosCampinaVerde = pd.read_csv(path + '/MG-CAMPINA VERDE.csv',sep=',')
dadosSorriso = pd.read_csv(path + '/MT-SORRISO.csv',sep=',')
dadosDiamante = pd.read_csv(path + '/PR-DIAMANTE DO NORTE.csv',sep=',')
dadosCampo = pd.read_csv(path + '/RS-CAMPOBOM.csv',sep=',')

dadosBarcelos = pd.read_csv(path + '/AM-BARCELOS.csv',sep=',')
dadosMacapa = pd.read_csv(path + '/AP-MACAPÁ.csv',sep=',')
dadosPetrolina = pd.read_csv(path + '/PE-PETROLINA.csv',sep=',')
dadosIrece = pd.read_csv(path + '/BA-IRÊCE.csv',sep=',')

listNames = ["Campo Bom - RS","Sorriso - MT","Diamante do Norte - PR"]#["Campina Verde - MG","Barcelos - AM", "Macapá - AP", "Petrolina - PE", "Irêce - BA"]
listaDados = [dadosCampo,dadosSorriso,dadosDiamante]#[dadosCampinaVerde,dadosBarcelos,dadosMacapa,dadosPetrolina,dadosIrece]



fields = ['City', 'TrainningSize','RMSE','Time', 'Epoch','Training_Loss', 'Training_Accuracy', 'Test_Loss', 'Test_Accuracy', 'RMSE_UNKNOW']

pos = ['umid_inst', 'pto_orvalho_inst', 'pressao', 'radiacao', 'vento_vel']

y_validation = dadosParaiso['temp_inst'].to_numpy()
X_validation= dadosParaiso.drop(['temp_inst'],axis = 1)
scaler = StandardScaler().fit(X_validation)
X_validation = scaler.transform(X_validation)

num_folds = 10
epochs = 3000
seed = 7
learning_rate = 1e-4
min_delta = 1e-4
patience = 30
batch_size = 16


kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
mlp_time=[]

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('.', end=' ')
        print(f"Epoch {epoch+1}: loss = {logs['loss']} rmse= {logs['root_mean_squared_error']}")
callback = MyCallback()
early_stopping = EarlyStopping(monitor= 'root_mean_squared_error', min_delta=min_delta, patience=patience)

contCity =0
for data in listaDados:
    city = listNames[contCity]

    y = data['temp_inst'].to_numpy()
    X = data.drop(['temp_inst'],axis = 1)
    X_o = data.drop(['temp_inst'],axis = 1).values
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
        with open(file_path + name + '_' + city + '.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)


    for name, model in pipelines:
        file_name = file_path + name + '_' + city

        for i, (train_index, test_index) in enumerate(kfold.split(X)):
            results=[]
            X_fold_train = X[train_index]
            y_fold_train = y[train_index]
            X_fold_test = X[test_index]
            y_fold_test = y[test_index]

            X_f_train = X_o[train_index]
            X_f_test = X_o[test_index]

            cont_batch = 0
            def custom_cost_function(y_true,y_pred):
                    # Parâmetros
                    R = np.mean(X_f_train[:,2]) *1000 #radiação em kj/m²
                    rho = 1.184
                    ca= 1012
                    sigma = 5.67e-8
                    T_pred= tf.squeeze(y_pred) + 273.15 #temperatura em celsius
                    T_true= tf.squeeze(y_true) + 273.15 #temperatura em celsius
                    alpha = 2.7e-3
                    gamma = 6.5e-4
                    d2T_dt2_pred = (alpha * R - gamma * sigma * T_pred**4) / (ca * rho)
                    d2T_dt2_true = (alpha * R - gamma * sigma * T_true**4) / (ca * rho)

                    correction_factor = tf.abs(d2T_dt2_true) - tf.abs(d2T_dt2_pred)
                    loss =tf.reduce_mean( tf.square(y_true - y_pred ) + d2T_dt2_pred)
                    return loss

            class HeatEquationLoss(tf.keras.losses.Loss):
              def __init__(self, name='heat_equation_loss', **kwargs):
                  super(HeatEquationLoss, self).__init__(name=name, **kwargs)

              def call(self, y_true, y_pred):
                  return custom_cost_function(y_true, y_pred)

            history= None
            num_epochs = None
            y_pred = None
            train_score = None
            test_score = None

            inicio = time.time()
            if name not in ["MLP", "PINN"]:
                print(">>>>"+name)
                model.fit(X_fold_train, y_fold_train)
                train_score = [None, model.score(X_fold_train, y_fold_train)] # Avalie o modelo
                test_score = [None, model.score(X_fold_test, y_fold_test),None ] # Avalie o modelo
                y_pred = model.predict(X_fold_test)
                y_unknow = model.predict(X_validation)
            else:
                model_rn = None
                if name == "MLP":
                    print(">>>>"+name)
                    model_rn = tese.build_model(X_fold_train.shape[1])
                    model_rn.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[RootMeanSquaredError()], optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
                if name == "PINN":
                    print(">>>>"+name)
                    model_rn = tese.build_model(X_fold_train.shape[1])
                    model_rn.compile(loss=HeatEquationLoss(), metrics=[RootMeanSquaredError()], optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

                history = model_rn.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size, validation_data=(X_fold_test, y_fold_test),  verbose=0, callbacks=[callback, early_stopping])
                train_score = model_rn.evaluate(X_fold_train, y_fold_train)  # Avalie o modelo
                test_score = model_rn.evaluate(X_fold_test, y_fold_test)
                y_pred = model_rn.predict(X_fold_test)
                y_unknow = model_rn.predict(X_validation)
                num_epochs = len(history.epoch)
                tese.plot_rmse(history, i,file_name,city,name,None)

            fim = time.time()

            mlp_time = fim - inicio
            print(f'Fold: {str(i)} - Localidade: {city}')
            results.append([city, str(i), sqrt(mean_squared_error(y_fold_test, y_pred)), mlp_time, num_epochs, train_score[0],train_score[1],test_score[0],test_score[1],sqrt(mean_squared_error(y_validation, y_unknow))])
            values = ','.join(str(v) for v in results)
            with open(file_name + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(results)
                print("List saved to CSV file successfully.")
    contCity += 1
