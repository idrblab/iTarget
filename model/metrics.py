import copy
import numpy as np
from sklearn import metrics
# import tensorflow as tf
import time
import psutil
import os

def reshape_tf2th( X):
    # N, W, H, C = X.shape
    # print(f'feamap shape before reshape: N, W, H, C = {N, W, H, C}')
    X = np.moveaxis(X,3,1)
    # X = np.transpose(X,(0,3,1,2))
    print(f'feamap shape after reshape: N, C, W, H = {X.shape}')
    return X

def to_categorical(num_classes, y):
    return np.eye(num_classes, dtype=np.long)[y]

def evaluate(y_true, y_pred):
    y_label = y_true[:,-1]
    y_score = y_pred[:,-1]
    # ROC, AUC
    fprs, tprs, thresholds_auc = metrics.roc_curve(y_label, y_score)
    auc = metrics.auc(fprs, tprs)
    # PRAUC
    pres, recs, thresholds_prc = metrics.precision_recall_curve(y_label, y_score)
    prauc = metrics.auc(recs, pres)
    av_prc = metrics.average_precision_score(y_label, y_score)
    # scores' label prediction by threshold
    threshold = 0.5
    label_pred = copy.deepcopy(y_score)
    label_pred = np.where(y_score >= threshold, np.ones_like(label_pred), label_pred)
    label_pred = np.where(y_score < threshold, np.zeros_like(label_pred), label_pred)
    # TN, FP, FN, TP
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_label, y_pred=label_pred, labels=[0,1]).ravel()
    # Model Evaluation
    acc = metrics.accuracy_score(y_label, label_pred)
    mcc = metrics.matthews_corrcoef(y_label, label_pred)
    precision = metrics.precision_score(y_label, label_pred)
    recall = metrics.recall_score(y_label, label_pred)
    f1 = metrics.f1_score(y_label, label_pred)
    specificity = tn/(fp+tn)
    sensitivity = tp/(tp+fn)
    return fprs, tprs, thresholds_auc, pres, recs, np.append(thresholds_prc, [1], axis=0), tn, fp, fn, tp, acc, auc, mcc, precision, recall, specificity, sensitivity, f1, prauc, av_prc


# class CLA_EarlyStoppingAndPerformance(tf.keras.callbacks.Callback):
#     def __init__(self, train_data, valid_data, MASK = -1, patience=5, criteria = 'acc_val', metric = 'ROC', last_avf = None, verbose = 0):
#         super(CLA_EarlyStoppingAndPerformance, self).__init__()
#         sp = ['loss_val', 'acc_val', 'auc_val']
#         assert criteria in sp, 'not support %s ! only %s' % (criteria, sp)
#         self.x, self.y  = train_data
#         self.x_val, self.y_val = valid_data
#         self.last_avf = last_avf
        
#         self.history = {'epoch':[], 'loss':[], 'loss_val':[], 'auc':[], 'auc_val':[], 'acc':[], 'acc_val':[], 'mcc':[], 'mcc_val':[], 'precision':[], 'precision_val':[], 'recall':[], 'recall_val':[], 'specificity':[], 'specificity_val':[], 'sensitivity':[], 'sensitivity_val':[], 'f1':[], 'f1_val':[], 'prauc':[], 'prauc_val':[], 'av_prc':[], 'av_prc_val':[]}
#         # , 'fprs':[], 'fprs_val':[], 'tprs':[], 'tprs_val':[], 'thresholds_auc':[], 'thresholds_auc_val':[], 'pres':[], 'pres_val':[], 'recs':[], 'recs_val':[], 'thresholds_prc':[], 'thresholds_prc_val':[]}
#         self.MASK = MASK
#         self.patience = patience
#         self.criteria = criteria
#         self.metric = metric
#         self.best_epoch = 0
#         self.verbose = verbose

#     def on_train_begin(self, logs=None):
#         self.time_ep = time.time()
#         # The number of epoch it has waited when loss is no longer minimum.
#         self.wait = 0
#         # The epoch the training stops at.
#         self.stopped_epoch = 0
#         # Initialize the best as infinity.
#         if self.criteria == 'loss_val':
#             self.best = np.Inf  
#         else:
#             self.best = -np.Inf
        
#     def on_epoch_end(self, epoch, logs={}):
#         print(u'memory used: %.2f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
#         time_cost = time.time() - self.time_ep
#         print('time for one epoch: ', f'{time_cost:.4f}')
#         self.time_ep = time.time()
#         y_pred = self.model.predict(self.x)
#         if self.last_avf == None:
#             y_pred = 1/(1+np.exp(-y_pred))
#         else:
#             y_pred = y_pred
#         # roc_list = self.roc_auc(self.y, y_pred)
#         # roc_mean = np.nanmean(roc_list)
#         fprs, tprs, thresholds_auc, pres, recs, thresholds_prc, tn, fp, fn, tp, acc, auc, mcc, precision, recall, specificity, sensitivity, f1, prauc, av_prc = evaluate(self.y, y_pred)
        
        
#         y_pred_val = self.model.predict(self.x_val)
#         # roc_val_list = self.roc_auc(self.y_val, y_pred_val)        
#         # roc_val_mean = np.nanmean(roc_val_list)
#         fprs_val, tprs_val, thresholds_auc_val, pres_val, recs_val, thresholds_prc_val, tn_val, fp_val, fn_val, tp_val, acc_val, auc_val, mcc_val, precision_val, recall_val, specificity_val, sensitivity_val, f1_val, prauc_val, av_prc_val = evaluate(self.y_val, y_pred_val)

        
#         self.history['epoch'].append(epoch)
#         self.history['loss'].append(logs.get('loss'))
#         self.history['loss_val'].append(logs.get('val_loss'))
#         # self.history['auc'].append(roc_mean)
#         # self.history['val_auc'].append(roc_val_mean)
#         self.history['auc'].append(auc)
#         self.history['auc_val'].append(auc_val)
#         self.history['acc'].append(acc)
#         self.history['acc_val'].append(acc_val)
#         self.history['mcc'].append(mcc)
#         self.history['mcc_val'].append(mcc_val)
#         self.history['precision'].append(precision)
#         self.history['precision_val'].append(precision_val)
#         self.history['recall'].append(recall)
#         self.history['recall_val'].append(recall_val)
#         self.history['specificity'].append(specificity)
#         self.history['specificity_val'].append(specificity_val)
#         self.history['sensitivity'].append(sensitivity)
#         self.history['sensitivity_val'].append(sensitivity_val)
#         self.history['f1'].append(f1)
#         self.history['f1_val'].append(f1_val)
#         self.history['prauc'].append(prauc)
#         self.history['prauc_val'].append(prauc_val)
#         self.history['av_prc'].append(av_prc)
#         self.history['av_prc_val'].append(av_prc_val)
#         # self.history'fprs':[]
#         # self.history'fprs_val':[]
#         # self.history'tprs':[]
#         # self.history'tprs_val':[]
#         # self.history'thresholds_auc':[]
#         # self.history'thresholds_auc_val':[]
#         # self.history'pres':[]
#         # self.history'pres_val':[]
#         # self.history'recs':[]
#         # self.history'recs_val':[]
#         # self.history'thresholds_prc':[]
#         # self.history'thresholds_prc_val':[]


#         eph = str(epoch+1).zfill(4)        
#         loss = '{0:.4f}'.format((logs.get('loss')))
#         loss_val = '{0:.4f}'.format((logs.get('val_loss')))
#         # auc = '{0:.4f}'.format(roc_mean)
#         # auc_val = '{0:.4f}'.format(roc_val_mean)    
        
#         if self.verbose:
#             print(f'---epoch{eph}---: loss: {loss} - loss_val: {loss_val} - acc_train: {acc:.8f} - acc_val: {acc_val:.8f} - auc_train: {auc:.8f} - auc_val: {auc_val:.8f} ')
#             print(f'acc_val = {acc_val:.4f}; auc_val = {auc_val:.4f}, mcc_val = {mcc_val:.4f}, precision_val = {precision_val:.4f}, recall_val = {recall_val:.4f}, specificity_val = {specificity_val:.4f}, sensitivity_val = {sensitivity_val:.4f}, f1_val = {f1_val:.4f}, prauc_val = {prauc_val:.4f}, av_prc_val = {av_prc_val:.4f}')

#         if self.criteria == 'loss_val':
#             current = logs.get('val_loss')
#             if current <= self.best:
#                 self.best = current
#                 self.wait = 0
#                 # Record the best weights if current results is better (less).
#                 self.best_weights = self.model.get_weights()
#                 self.best_epoch = epoch
#             else:
#                 self.wait += 1
#                 if self.wait >= self.patience:
#                     self.stopped_epoch = epoch
#                     self.model.stop_training = True
#                     print('\nRestoring model weights from the end of the best epoch.')
#                     self.model.set_weights(self.best_weights)    
#         elif self.criteria == 'acc_val':
#             current = acc_val
#             if current >= self.best:
#                 self.best = current
#                 self.wait = 0
#                 # Record the best weights if current results is better (less).
#                 self.best_weights = self.model.get_weights()
#                 self.best_epoch = epoch
#             else:
#                 self.wait += 1
#                 if self.wait >= self.patience:
#                     self.stopped_epoch = epoch
#                     self.model.stop_training = True
#                     print('\nRestoring model weights from the end of the best epoch.')
#                     self.model.set_weights(self.best_weights)  
#         else:
#             current = auc_val
#             if current >= self.best:
#                 self.best = current
#                 self.wait = 0
#                 # Record the best weights if current results is better (less).
#                 self.best_weights = self.model.get_weights()
#                 self.best_epoch = epoch
#             else:
#                 self.wait += 1
#                 if self.wait >= self.patience:
#                     self.stopped_epoch = epoch
#                     self.model.stop_training = True
#                     print('\nRestoring model weights from the end of the best epoch.')
#                     self.model.set_weights(self.best_weights)              
    
#     def on_train_end(self, logs=None):
#         self.model.set_weights(self.best_weights)
#         if self.stopped_epoch > 0:
#             print('\nEpoch %05d: early stopping' % (self.stopped_epoch + 1))
     
#     def evaluate(self, testX, testY):
        
#         y_pred = self.model.predict(testX)
#         roc_list = self.roc_auc(testY, y_pred)
#         return roc_list        


# from sklearn.metrics import roc_auc_score, precision_recall_curve
# from sklearn.metrics import auc as calculate_auc
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# def prc_auc_score(y_true, y_score):
#     precision, recall, threshold  = precision_recall_curve(y_true, y_score) #PRC_AUC
#     auc = calculate_auc(recall, precision)
#     return auc
# class CLA_EarlyStoppingAndPerformance_o(tf.keras.callbacks.Callback):

#     def __init__(self, train_data, valid_data, MASK = -1, patience=5, criteria = 'val_loss', metric = 'ROC', last_avf = None, verbose = 0):
#         super(CLA_EarlyStoppingAndPerformance, self).__init__()
        
#         sp = ['val_loss', 'val_auc']
#         assert criteria in sp, 'not support %s ! only %s' % (criteria, sp)
#         self.x, self.y  = train_data
#         self.x_val, self.y_val = valid_data
#         self.last_avf = last_avf
        
#         self.history = {'loss':[],
#                         'val_loss':[],
#                         'auc':[],
#                         'val_auc':[],
                        
#                         'epoch':[]}
#         self.MASK = MASK
#         self.patience = patience
#         # best_weights to store the weights at which the minimum loss occurs.
#         self.best_weights = None
#         self.criteria = criteria
#         self.metric = metric
#         self.best_epoch = 0
#         self.verbose = verbose
        
#     def sigmoid(self, x):
#         s = 1/(1+np.exp(-x))
#         return s

    
#     def roc_auc(self, y_true, y_pred):
#         if self.last_avf == None:
#             y_pred_logits = self.sigmoid(y_pred)
#         else:
#             y_pred_logits = y_pred
            
#         N_classes = y_pred_logits.shape[1]

#         aucs = []
#         for i in range(N_classes):
#             y_pred_one_class = y_pred_logits[:,i]
#             y_true_one_class = y_true[:, i]
#             mask = ~(y_true_one_class == self.MASK)
#             try:
#                 if self.metric == 'ROC':
#                     auc = roc_auc_score(y_true_one_class[mask], y_pred_one_class[mask]) #ROC_AUC
#                 elif self.metric == 'PRC': 
#                     auc = prc_auc_score(y_true_one_class[mask], y_pred_one_class[mask]) #PRC_AUC
#                 elif self.metric == 'ACC':
#                     auc = accuracy_score(y_true_one_class[mask], np.round(y_pred_one_class[mask])) #ACC
#                 elif self.metric == 'all':
#                     tn, fp, fn, tp = confusion_matrix(y_true=y_true_one_class[mask], y_pred=np.round(y_pred_one_class[mask]), labels=[0,1]).ravel()
#                     auc = (tn, fp, fn, tp)
#             except:
#                 auc = np.nan
#             aucs.append(auc)
#         return aucs  
    
        
        
#     def on_train_begin(self, logs=None):
#         # The number of epoch it has waited when loss is no longer minimum.
#         self.wait = 0
#         # The epoch the training stops at.
#         self.stopped_epoch = 0
#         # Initialize the best as infinity.
#         if self.criteria == 'val_loss':
#             self.best = np.Inf  
#         else:
#             self.best = -np.Inf
            

        
 
        
#     def on_epoch_end(self, epoch, logs={}):
        
#         y_pred = self.model.predict(self.x)
#         roc_list = self.roc_auc(self.y, y_pred)
#         roc_mean = np.nanmean(roc_list)
        
#         y_pred_val = self.model.predict(self.x_val)
#         roc_val_list = self.roc_auc(self.y_val, y_pred_val)        
#         roc_val_mean = np.nanmean(roc_val_list)
        
#         self.history['loss'].append(logs.get('loss'))
#         self.history['val_loss'].append(logs.get('val_loss'))
#         self.history['auc'].append(roc_mean)
#         self.history['val_auc'].append(roc_val_mean)
#         self.history['epoch'].append(epoch)
        
        
#         eph = str(epoch+1).zfill(4)        
#         loss = '{0:.4f}'.format((logs.get('loss')))
#         val_loss = '{0:.4f}'.format((logs.get('val_loss')))
#         auc = '{0:.4f}'.format(roc_mean)
#         auc_val = '{0:.4f}'.format(roc_val_mean)    
        
#         if self.verbose:
#             if self.metric == 'ACC':
#                 print('\repoch: %s, loss: %s - val_loss: %s; acc: %s - val_acc: %s' % (eph,
#                                                                                    loss, 
#                                                                                    val_loss, 
#                                                                                    auc,
#                                                                                    auc_val), end=100*' '+'\n')
#             elif self.metric == 'all':
#                 print('\repoch: %s, loss: %s - val_loss: %s; acc: %s - val_acc: %s' % (eph,
#                                                                                    loss, 
#                                                                                    val_loss, 
#                                                                                    auc,
#                                                                                    auc_val), end=100*' '+'\n')
#             else:
#                 print('\repoch: %s, loss: %s - val_loss: %s; auc: %s - val_auc: %s' % (eph,
#                                                                                    loss, 
#                                                                                    val_loss, 
#                                                                                    auc,
#                                                                                    auc_val), end=100*' '+'\n')


#         if self.criteria == 'val_loss':
#             current = logs.get(self.criteria)
#             if current <= self.best:
#                 self.best = current
#                 self.wait = 0
#                 # Record the best weights if current results is better (less).
#                 self.best_weights = self.model.get_weights()
#                 self.best_epoch = epoch

#             else:
#                 self.wait += 1
#                 if self.wait >= self.patience:
#                     self.stopped_epoch = epoch
#                     self.model.stop_training = True
#                     print('\nRestoring model weights from the end of the best epoch.')
#                     self.model.set_weights(self.best_weights)    
                    
#         else:
#             current = roc_val_mean
#             if current >= self.best:
#                 self.best = current
#                 self.wait = 0
#                 # Record the best weights if current results is better (less).
#                 self.best_weights = self.model.get_weights()
#                 self.best_epoch = epoch

#             else:
#                 self.wait += 1
#                 if self.wait >= self.patience:
#                     self.stopped_epoch = epoch
#                     self.model.stop_training = True
#                     print('\nRestoring model weights from the end of the best epoch.')
#                     self.model.set_weights(self.best_weights)              
    
#     def on_train_end(self, logs=None):
#         self.model.set_weights(self.best_weights)
#         if self.stopped_epoch > 0:
#             print('\nEpoch %05d: early stopping' % (self.stopped_epoch + 1))

        
#     def evaluate(self, testX, testY):
        
#         y_pred = self.model.predict(testX)
#         roc_list = self.roc_auc(testY, y_pred)
#         return roc_list            

 