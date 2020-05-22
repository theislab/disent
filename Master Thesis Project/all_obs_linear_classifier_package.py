import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import glob, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


def clean_input(data,dim_size):
    # converts the input data to columns for every dimension
    col_names = []  

    for i in range(dim_size):
        col_names.append("dim"+str(i))
    
    df_list = []
    for i in range(len(data)):
        element = data[i]
        element_eval = eval(element)
        df_list.append(element_eval[0])
    
    df = pd.DataFrame(df_list,columns=col_names)
    return df

def nll1(y_true, y_pred):
    # loss function for model
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

def create_model(output_dim ,input_dim = 1):
    # create model
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    # Compile model
    model.compile(loss=nll1, optimizer='adagrad', metrics=['accuracy'])
    return model

def combine_batch_files(observation):
    #combining all files generated for scores
    string = observation+"_disentangled_score/matrix_all_dim"
    all_filenames = [i for i in glob.glob(string+"[0-9]*.csv")]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    combined_csv = combined_csv.reset_index(drop=True)
    return combined_csv

def feature_classification(path,z_dim,observation):
    os.chdir(path)
    dataframe = combine_batch_files(observation)
    print(dataframe)
    dentate_acc = feature_keras_linear_classifier(dataframe,z_dim)
    dentate_acc.to_csv(observation+"keras_linear_classifier_output_local.csv")

def feature_keras_linear_classifier(df,z_dim):

    X = df.iloc[:,1]

    x_df = clean_input(X,z_dim)

    Y = df.iloc[:,0]
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    un_lab = list(set(encoded_Y))
    y_labels = encoder.inverse_transform(un_lab)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    output_dim = len(y_labels)
    
    results_all_dim = []
    for dim in range(z_dim):
        print("--------------------------------------------------------------",str(dim))
        x_dim = x_df.iloc[:,dim]
        x_train, x_test, y_train, y_test = train_test_split(x_dim, dummy_y,test_size=0.30)
        model = create_model(output_dim)
        model.fit(x_train, y_train,epochs=50, batch_size=10, verbose=1)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        print(classification_report(y_test,y_pred))
        print(confusion_matrix(y_test,y_pred))
        conf_mat = confusion_matrix(y_test,y_pred)
        # calculating accuracy per feature
        acc = conf_mat.diagonal()/np.sum(conf_mat,axis=0)
        print(acc)
        acc = list(acc)
        results_all_dim.append(acc)


    results_all_dim =  pd.DataFrame(results_all_dim,columns=y_labels)
    return results_all_dim

    

