
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas  as pd
import warnings
import pickle
from sklearn.svm import SVC 

df_features = pd.DataFrame()
final_Train = pd.DataFrame()


# In[10]:


def getPloyFit(Df_Meal_Data):
    
    for i in range(len(Df_Meal_Data)):
      time_Stamps= [i for i in range(len(Df_Meal_Data.iloc[i]))]
      times = np.array(time_Stamps)
      coefs = np.polyfit(times,Df_Meal_Data.iloc[i], 4)
      coefs1 = np.polyfit(times,Df_Meal_Data.iloc[i], 2)
      coefs2 = np.polyfit(times,Df_Meal_Data.iloc[i], 3)
      coefs3 = np.polyfit(times,Df_Meal_Data.iloc[i], 1)

      df_features.at[i,'polyFit_4Degree_First_Coef']=coefs[0]
      df_features.at[i,'polyFit_4Degree_Sec_Coef']=coefs[1]
      df_features.at[i,'polyFit_4Degree_Third_Coef']=coefs[2]
      df_features.at[i,'polyFit_4Degree_Fourth_Coef']=coefs[3]
      df_features.at[i,'polyFit_4Degree_Five_Coef']=coefs[4]
      df_features.at[i,'polyFit_2Degree_First_Coef']=coefs1[0]
      df_features.at[i,'polyFit_2Degree_Sec_Coef']=coefs1[1]
      df_features.at[i,'polyFit_2Degree_Third_Coef']=coefs1[2]
      df_features.at[i,'polyFit_3Degree_First_Coef']=coefs2[0]
      df_features.at[i,'polyFit_3Degree_Sec_Coef']=coefs2[1]
      df_features.at[i,'polyFit_3Degree_Third_Coef']=coefs2[2]
      df_features.at[i,'polyFit_3Degree_Fourth_Coef']=coefs2[3]
      df_features.at[i,'polyFit_1Degree_First_Coef']=coefs3[0]
      df_features.at[i,'polyFit_1Degree_Sec_Coef']=coefs3[1]



def getPST(Df_Meal_Data):
    for i in range(len(Df_Meal_Data)):
        ps = np.abs(np.fft.fft(Df_Meal_Data.iloc[i])) ** 2
        ps.sort()
        values = ps[:-8]
        df_features.at[i, 'PST_First_Peak'] = values[7]
        df_features.at[i, 'PST_Sec_Peak'] = values[6]
        df_features.at[i, 'PST_Third_Peak'] = values[5]
        df_features.at[i, 'PST_Fourth_Peak'] = values[4]
        df_features.at[i, 'PST_Fifth_Peak'] = values[3]
        df_features.at[i, 'PST_Sixth_Peak'] = values[2]
        df_features.at[i, 'PST_seventh_Peak'] = values[1]
        df_features.at[i, 'PST_eight_Peak'] = values[0]

def getFFT(Df_Meal_Data):
    for i in range(len(Df_Meal_Data)):
        ps = np.abs(np.fft.fft(Df_Meal_Data.iloc[i]))
        ps.sort()
        values = ps[:-8]
        df_features.at[i, 'FFT_First_Peak'] = values[7]
        df_features.at[i, 'FFT_Sec_Peak'] = values[6]
        df_features.at[i, 'FFT_Third_Peak'] = values[5]
        df_features.at[i, 'FFT_Fourth_Peak'] = values[4]
        df_features.at[i, 'FFT_Fifth_Peak'] = values[3]
        df_features.at[i, 'FFT_Sixth_Peak'] = values[2]
        df_features.at[i, 'FFT_seventh_Peak'] = values[1]
        df_features.at[i, 'FFT_eight_Peak'] = values[0]




def autocorrelationFeatures(Df_Meal_Data):
    for i in range(len(Df_Meal_Data)):
      s = pd.Series(Df_Meal_Data.iloc[i]).autocorr(lag=2)
      q = pd.Series(Df_Meal_Data.iloc[i]).autocorr(lag=3)
      p = pd.Series(Df_Meal_Data.iloc[i]).autocorr(lag=5)
      s1 = pd.Series(Df_Meal_Data.iloc[i]).autocorr(lag=4)
      q1 = pd.Series(Df_Meal_Data.iloc[i]).autocorr(lag=6)
      p1 = pd.Series(Df_Meal_Data.iloc[i]).autocorr(lag=8)
      df_features.at[i,'Correlation_With_Lag_2']=s
      df_features.at[i,'Correlation_With_Lag_3']=q
      df_features.at[i,'Correlation_With_Lag_5']=p
      df_features.at[i,'Correlation_With_Lag_4']=s1
      df_features.at[i,'Correlation_With_Lag_6']=q1
      df_features.at[i,'Correlation_With_Lag_8']=p1


def zerocrossings(Df_Meal_Data):
    for i in range(len(Df_Meal_Data)):
      zero_crossings = np.where(np.diff(np.sign(Df_Meal_Data.iloc[i])))[0]
      df_features.at[i,'number_of_zero_crossings']=len(zero_crossings)

def pcaOfTheFeatures():
    df_features.fillna(value=0,inplace=True)
    eigen_vectors_top=np.loadtxt("eigen_vectors_top.csv", delimiter=",")
    pca_of_the_data = np.dot(df_features, eigen_vectors_top.T)
    return pca_of_the_data


def predictData():
    final_Train = pcaOfTheFeatures()
    #np.savetxt("final_Train.csv",final_Train,delimiter=",")
    filename = 'SVM.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    svm_predictions = loaded_model.predict(final_Train)
    svm_predictions=svm_predictions.astype(int)
    print(svm_predictions)
    np.savetxt("Amarnath_Tadigadapa.csv", svm_predictions)
    filename = 'LogisticReg.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    log_predictions = loaded_model.predict(final_Train)
    log_predictions = log_predictions.astype(int)
    print(log_predictions)
    np.savetxt("Sunil_Samal.csv", log_predictions)
    filename = 'NaivesBayes.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    NB_predictions = loaded_model.predict(final_Train)
    NB_predictions = NB_predictions.astype(int)
    print(NB_predictions)
    np.savetxt("Arijit_panda.csv", NB_predictions)
    filename = 'perceptron.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    MLPClassifier_predictions = loaded_model.predict(final_Train)
    MLPClassifier_predictions = MLPClassifier_predictions.astype(int)
    print(MLPClassifier_predictions)
    np.savetxt("Asmi_Pattnaik.csv", MLPClassifier_predictions)

def main():
    arguments = len(sys.argv) - 1
    file_name = sys.argv[arguments]
    data_frame = pd.read_csv(file_name,header=None)
    data_frame.fillna(value=0,inplace=True)
    getFFT(data_frame)
    getPloyFit(data_frame)
    autocorrelationFeatures(data_frame)
    zerocrossings(data_frame)
    getPST(data_frame)
    predictData()


if __name__ == "__main__":
    main()





