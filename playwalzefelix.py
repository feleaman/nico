#++++++++++Import Modules++++++++++
import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC, OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, average_precision_score
from datetime import datetime
from argparse import ArgumentParser
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from m_plots import *

#++++++++++Inputs++++++++++
mypath = 'M:\Schneidtechnik\Rotationsprüfstand\Messdaten\Walze4.0\Python_Analysis\Data\Int_Analysis_all_20201002_Threshold_300mV.csv'

Inputs = ['mode']
InputsOpt_Defaults = {'rs':11, 'test_size':0.2, 'cv':10, 'path':mypath, 'stratify':True, 'scaler':True, 'pca':None}

def main(argv):
    config = read_parser(argv, Inputs, InputsOpt_Defaults)

    if config['mode'] == 'test':
        
        
        print('test')
    
    elif config['mode'] == 'learn_svm':        
        #+++Load data
        if config['path'] == None:
            root = Tk()
            root.withdraw()
            root.update()
            filepath = filedialog.askopenfilename()
            root.destroy()    
            filename = os.path.basename(filepath)
        else:
            filepath = config['path']
            filename = os.path.basename(filepath)
        
        
        #+++Construct features matrix and label vector
        myDF = pd.read_csv(filepath)
        mydict = myDF.to_dict(orient='list')
        n = len(mydict['Label'])
        y = np.zeros(n)
        y = [int(y[i]) if mydict['Label'][i]=='Concrete' else 1 for i in range(n)]
        X = []
        Features = ['Area_under_curve', 'Crest-Factor', 'Energy', 'Kurtosis', 'Peak_amplitude', 'RMS', 'Ring_down', 'Signal_strength', 'Skewnes', 'StDev', 'Variance']
        k = 0
        for key in mydict.keys():
            if key in Features:
                X.append(mydict[key])
            k += 1
        X = np.array(X)
        X = np.transpose(X)
        
        
        #+++Train/Test Split
        if config['stratify'] == True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['rs'], stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['rs'])
           
           
        #+++Scaler
        if config['scaler'] == True:
            print('With standard scaler')
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        
        #+++PCA
        if config['pca'] != None:
            pca = PCA(n_components=config['pca'])
            pca.fit(X_train)
            print('PCA results: ', pca.explained_variance_ratio_)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
        
        
        #+++Hyperparameters for Grid search
        Penalizations = [0.1, 1.0, 10.]
        Kernels = ['linear', 'rbf', 'poly']

        results = {}
        results['penal'] = []
        results['kernel'] = []
        
        results['accu_cv'] = []
        results['accu_te'] = []
        results['recall_cv'] = []
        results['recall_te'] = []
        results['preci_cv'] = []
        results['preci_te'] = []
        results['f1_cv'] = []
        results['f1_te'] = []
        
        
        count = 0
        for kernel_ in Kernels:        
            for penal_ in Penalizations:
                print('+++++++Case = ', count)
                clf = SVC(kernel=kernel_, C=penal_, gamma='auto', verbose=False, max_iter=100000, random_state=config['rs'])
                

                
                scores = cross_validate(clf, X_train, y_train, cv=config['cv'], scoring=('accuracy', 'recall', 'precision', 'f1'))

                
                clf.fit(X_train, y_train)
                Pred_y_test = clf.predict(X_test)
                
                
                score_test_accu = accuracy_score(y_test, Pred_y_test)
                score_test_recall = recall_score(y_test, Pred_y_test)
                score_test_preci = precision_score(y_test, Pred_y_test)
                score_test_f1 = f1_score(y_test, Pred_y_test)
                

                results['penal'].append(penal_)
                results['kernel'].append(kernel_)
                
                results['accu_cv'].append(scores['test_accuracy'].mean())
                results['recall_cv'].append(scores['test_recall'].mean())
                results['preci_cv'].append(scores['test_precision'].mean())
                results['f1_cv'].append(scores['test_f1'].mean())
                
                
                results['accu_te'].append(score_test_accu)
                results['recall_te'].append(score_test_recall)
                results['preci_te'].append(score_test_preci)
                results['f1_te'].append(score_test_f1)
                
                count +=1
                
        
        
        #+++Save results
        config['features'] = Features
        config['filename'] = filename        

        name = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(name)
        save_pickle('config_' + name + '.pkl', config)    
        
        DataFr = pd.DataFrame(data=results, index=None)        
        with pd.ExcelWriter('results_' + name + '.xlsx') as writer:
            DataFr.to_excel(writer, sheet_name='SVM_Learn')        
        print('Result OK')
        
        
    
    
    elif config['mode'] == 'learn_oneclass':        
        #+++Load data
        if config['path'] == None:
            root = Tk()
            root.withdraw()
            root.update()
            filepath = filedialog.askopenfilename()
            root.destroy()    
            filename = os.path.basename(filepath)
        else:
            filepath = config['path']
            filename = os.path.basename(filepath)
        
        
        #+++Construct features matrix and label vector
        myDF = pd.read_csv(filepath)
        mydict = myDF.to_dict(orient='list')
        n = len(mydict['Label'])
        y = np.ones(n)
        y = [int(y[i]) if mydict['Label'][i]=='Concrete' else -1 for i in range(n)]

        X = []
        Features = ['Area_under_curve', 'Crest-Factor', 'Energy', 'Kurtosis', 'Peak_amplitude', 'RMS', 'Ring_down', 'Signal_strength', 'Skewnes', 'StDev', 'Variance']
        k = 0
        for key in mydict.keys():
            if key in Features:
                X.append(mydict[key])
            k += 1
        X = np.array(X)
        X = np.transpose(X)
        
        
        #+++Train/Test Split
        if config['stratify'] == True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['rs'], stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['rs'])
           
           
        #+++Scaler
        if config['scaler'] == True:
            print('With standard scaler')
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        
        #+++PCA
        if config['pca'] != None:
            pca = PCA(n_components=config['pca'])
            pca.fit(X_train)
            print('PCA results: ', pca.explained_variance_ratio_)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
        
        
        #+++Hyperparameters for Grid search
        Nus = [0.1, 0.5, 0.9]
        Kernels = ['linear', 'rbf', 'poly']

        results = {}
        results['nu'] = []
        results['kernel'] = []
        
        results['accu_cv'] = []
        results['accu_te'] = []
        results['recall_cv'] = []
        results['recall_te'] = []
        results['preci_cv'] = []
        results['preci_te'] = []
        results['f1_cv'] = []
        results['f1_te'] = []
        results['Baccu_cv'] = []
        results['Baccu_te'] = []
        results['Bpreci_cv'] = []
        results['Bpreci_te'] = []
        
        
        count = 0
        for kernel_ in Kernels:        
            for nu_ in Nus:
                print('+++++++Case = ', count)
                clf = OneClassSVM(kernel=kernel_, nu=nu_, gamma='auto', verbose=False, max_iter=100000)        
                

                # scores = cross_validate(clf, X_train, y_train, cv=config['cv'], scoring=('accuracy', 'recall', 'precision', 'f1'))
                scores = cross_validate(clf, X_train, y_train, cv=config['cv'], scoring=('accuracy', 'recall', 'precision', 'f1', 'balanced_accuracy', 'average_precision'))
                
                clf.fit(X_train, y_train)
                Pred_y_test = clf.predict(X_test)
                
                
                score_test_accu = accuracy_score(y_test, Pred_y_test)
                score_test_recall = recall_score(y_test, Pred_y_test)
                score_test_preci = precision_score(y_test, Pred_y_test)
                score_test_f1 = f1_score(y_test, Pred_y_test)
                score_test_Baccu = balanced_accuracy_score(y_test, Pred_y_test)
                score_test_Bpreci = average_precision_score(y_test, Pred_y_test)
                

                # results['penal'].append(penal_)
                results['nu'].append(nu_)
                results['kernel'].append(kernel_)
                
                results['accu_cv'].append(scores['test_accuracy'].mean())
                results['recall_cv'].append(scores['test_recall'].mean())
                results['preci_cv'].append(scores['test_precision'].mean())
                results['f1_cv'].append(scores['test_f1'].mean())
                results['Baccu_cv'].append(scores['test_balanced_accuracy'].mean())
                results['Bpreci_cv'].append(scores['test_average_precision'].mean())
                
                
                results['accu_te'].append(score_test_accu)
                results['recall_te'].append(score_test_recall)
                results['preci_te'].append(score_test_preci)
                results['f1_te'].append(score_test_f1)
                results['Baccu_te'].append(score_test_Baccu)
                results['Bpreci_te'].append(score_test_Bpreci)
                
                count +=1
                
        
        
        #+++Save results
        config['features'] = Features
        config['filename'] = filename        

        name = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(name)
        save_pickle('config_' + name + '.pkl', config)    
        
        DataFr = pd.DataFrame(data=results, index=None)        
        with pd.ExcelWriter('results_' + name + '.xlsx') as writer:
            DataFr.to_excel(writer, sheet_name='SVM_Learn')        
        print('Result OK')

        
        
    else:
        print('error mode')
        
        
        
        
    return

def read_parser(argv, Inputs, InputsOpt_Defaults):
    Inputs_opt = [key for key in InputsOpt_Defaults]
    Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
    parser = ArgumentParser()
    for element in (Inputs + Inputs_opt):
        print(element)
        if element == 'no_element':
            parser.add_argument('--' + element, nargs='+')
        else:
            parser.add_argument('--' + element, nargs='?')
    
    args = parser.parse_args()
    config = {}
    for element in Inputs:
        if getattr(args, element) != None:
            config[element] = getattr(args, element)
        else:
            print('Required:', element)
            sys.exit()

    for element, value in zip(Inputs_opt, Defaults):
        if getattr(args, element) != None:
            config[element] = getattr(args, element)
        else:
            print('Default ' + element + ' = ', value)
            config[element] = value
    
    #Type conversion to float
    # if config['power2'] != 'auto' and config['power2'] != 'OFF':
        # config['power2'] = int(config['power2'])
    # config['fs_tacho'] = float(config['fs_tacho'])
    # config['fs_signal'] = float(config['fs_signal'])
    config['test_size'] = float(config['test_size'])
    # config['penalty'] = float(config['penalty'])
    config['cv'] = int(config['cv'])
    if config['rs'] != None:
        config['rs'] = int(config['rs'])
        
    if config['pca'] != None:
        config['pca'] = int(config['pca'])
    #Type conversion to int    
    # Variable conversion
    return config

if __name__ == '__main__':
    main(sys.argv)







sys.exit()







##############################################################
#+++++++++++++++++++++++++++Data Loading
mypath = 'M:\Schneidtechnik\Rotationsprüfstand\Messdaten\Walze4.0\Python_Analysis\Data\Int_Analysis_all_20201002_Threshold_300mV.csv'
myDF = pd.read_csv(mypath)
mydict = myDF.to_dict(orient='list')
n = len(mydict['Label'])
y = np.zeros(n)
y = [int(y[i]) if mydict['Label'][i]=='Concrete' else 1 for i in range(n)]
X = []
Features = ['Area_under_curve', 'Crest-Factor', 'Energy', 'Kurtosis', 'Peak_amplitude', 'RMS', 'Ring_down', 'Signal_strength', 'Skewnes', 'StDev', 'Variance']
k = 0
for key in mydict.keys():
    if key in Features:
        X.append(mydict[key])
        #X[:,k] = list(mydict[key])
    k += 1
X = np.array(X)
X = np.transpose(X)


# for key in mydict.keys():
    # print(key)
# pos = 0
# for i in y:
    # if i == 1:
        # pos +=1
# print(pos/len(y))




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)




scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# pca = PCA(n_components=3)
# pca.fit(X_train)
# print(pca.explained_variance_ratio_)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)



# Penalizations = [0.1, 1.0, 100., 1000., 10000.]
Penalizations = [0.1, 1.0, 10.]
Kernels = ['linear', 'poly', 'rbf']
# Kernels = ['linear']

results = {}
results['f1_va_mean'] = []
results['f1_va_std'] = []
results['f1_te'] = []
results['penal'] = []
results['kernel'] = []
count = 0
for kernel_ in Kernels:
    for penal_ in Penalizations:
        print(count)
        clf_1 = SVC(kernel=kernel_, C=penal_, gamma='auto', verbose=True, max_iter=-1, random_state=13, shrinking=False)
        
        scores = cross_val_score(clf_1, X_train, y_train, cv=10, scoring='f1')
        
        score_mean = scores.mean()
        score_std = scores.std()
        
        clf_1.fit(X_train, y_train)
        Pred_y_test = clf_1.predict(X_test)

        score_test = f1_score(y_test, Pred_y_test)

        results['f1_va_mean'].append(score_mean)
        results['f1_va_std'].append(score_std)
        results['f1_te'].append(score_test)
        results['penal'].append(penal_)
        results['kernel'].append(kernel_)
        count += 1
        



# if config['save'] == 'ON':
    # config['features'] = mykeys
    # config['filename'] = filename
    
    
    # print(datetime.now())
    # name = str(datetime.now())
name = datetime.now().strftime("%Y%m%d_%H%M%S")

save_pickle('config_' + name + '.pkl', results)        
DataFr = pd.DataFrame(data=results, index=None)

with pd.ExcelWriter('results_' + name + '.xlsx') as writer:
    DataFr.to_excel(writer, sheet_name='SVM_Learn')    


# writer = pd.ExcelWriter('results_' + name + '.xlsx')            
# DataFr.to_excel(writer, sheet_name='SVM_Learn')    



print('Result OK!')




# # In[93]:


# clf = SVC(gamma='auto', verbose=False, max_iter=500)
# #clf = SVC(kernel='linear', C=0.5, gamma='auto', verbose=False, max_iter=500)

# #clf.fit(X_train, y_train)


# # In[85]:


# scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1', verbose=0)
# score_mean = scores.mean()
# score_std = scores.std()
# print(score_mean)
# print(score_std)


# # In[94]:


# scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy', verbose=0)
# score_mean = scores.mean()
# score_std = scores.std()
# print(score_mean)
# print(score_std)


# # In[37]:


# # clf = SVC(kernel='linear', C=0.1, gamma='auto', verbose=False, max_iter=100000)
# # clf.fit(X_train, y_train)
# # scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1')
# # print('\n+++Validation Score: ', scores)


# # In[87]:


# scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='precision', verbose=0)
# score_mean = scores.mean()
# score_std = scores.std()
# print(score_mean)
# print(score_std)


# # In[95]:


# scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='recall', verbose=0)
# score_mean = scores.mean()
# score_std = scores.std()
# print(score_mean)
# print(score_std)


# # In[88]:


# scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='specificity', verbose=0)
# score_mean = scores.mean()
# score_std = scores.std()
# print(score_mean)
# print(score_std)


# # In[ ]:




