# Data Manipulation
import numpy as nmpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import math
import operator
from scipy.special import expit
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

dbutils.widgets.text("input_link", "s3://6350-bd-an-assignment-2/project/all_stocks_5yr.csv") # s3://6350-bd-an-assignment-2/project/all_stocks_5yr.csv
input_link = dbutils.widgets.get("input_link")
df = spark.read.option("header", "true").option("inferSchema", "false").csv(input_link)
display(df)

# CSV options
# infer_schema = "false"
# first_row_is_header = "true"
# delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
# df = spark.read.format(file_type) \
#   .option("inferSchema", infer_schema) \
#   .option("header", first_row_is_header) \
#   .option("sep", delimiter) \
#   .load(file_location)

data = df.toPandas()
data = data[['date','open', 'high', 'low','close']]
data['open'] = data['open'].astype(float)
data['high'] = data['high'].astype(float)
data['low'] = data['low'].astype(float)
data['close'] = data['close'].astype(float)
data = data.groupby(['date'],as_index=False).mean()
def toDateTime(d):
    split = d.split('-')
    return datetime.datetime(year=int(split[0]), month=int(split[1]), day=int(split[2]))

data['date'] = data['date'].apply(toDateTime)
data.index = data.pop('date')
print(data)
data['close-open']= data.close - data.open
data['high-low']  = data.high - data.low
data =data.dropna()
display(data)
#Euclidean Distance
def eucledian(p1,p2):
    return math.sqrt(pow((p1[0] - p2[0]), 2))

def getKNeighbors(X, p2, k):
    distance_list = []
    for index, row in X.iterrows():
        distance = eucledian(row,p2)
        distance_list.append((row, distance))
    
    distance_list.sort(key=operator.itemgetter(1))
    k_neighbors = []
    for i in range(k):
        k_neighbors.append(distance_list[i][0])
    return k_neighbors
 
#Function to calculate KNN
def predict(X, y, k):
    correct = 0
    incorrect = 0
    
    for index, row in y.iterrows():
        k_neighbors = getKNeighbors(X, row, k)
        negative = 0
        positive = 0
        for j in range(len(k_neighbors)):
            if k_neighbors[j][-2]>=0:
                positive +=1
            else:
                negative +=1
                
        if positive>=negative:
            if row[-2] >= 0:
                correct +=1
            else:
                incorrect +=1
        else:
            if row[-2] >= 0:
                incorrect +=1
            else:
                correct +=1
    
    return (correct/(correct + incorrect))

train, test = train_test_split(data,test_size=0.2, random_state=23)
dbutils.widgets.text("k_value", "7") # 7
k_value = int(dbutils.widgets.get("k_value"))
accuracy_test = predict(train, test, k_value)
print ('Test_data Accuracy: %.2f' %accuracy_test)
accuracy_train = predict(train, train, k_value)
print ('Train_data Accuracy: %.2f' %accuracy_train)
    
class LSTM:
    def __init__(self, hidden_neurons, input_size, learning_rate):
        self.hidden_neurons = hidden_neurons
        self.cells = input_size 
        self.lr_rate = learning_rate
        
        # gate states storage
        self.frgt_gt_stat = [nmpy.zeros((hidden_neurons,1)) for _ in range(self.cells)]
        self.ipt_gt_stat = [nmpy.zeros((hidden_neurons,1)) for _ in range(self.cells)]
        self.state_of_cand_gate = [nmpy.zeros((hidden_neurons,1)) for _ in range(self.cells)]
        self.c_state = [nmpy.zeros((hidden_neurons,1)) for _ in range(self.cells)] 
        self.o_p_gt_stat = [nmpy.zeros((hidden_neurons,1)) for _ in range(self.cells)]
        self.hid_gt_stat = [nmpy.zeros((hidden_neurons,1)) for _ in range(self.cells)] 
        
        # weights of gates
        self.wt_f = nmpy.random.random((self.hidden_neurons, self.hidden_neurons+1))/nmpy.sqrt(self.hidden_neurons+1)
        self.weight_of_inpt= nmpy.random.random((self.hidden_neurons, self.hidden_neurons+1))/nmpy.sqrt(self.hidden_neurons+1)
        self.weigt_of_c = nmpy.random.random((self.hidden_neurons, self.hidden_neurons+1))/nmpy.sqrt(self.hidden_neurons+1)
        self.weightt_otpt = nmpy.random.random((self.hidden_neurons, self.hidden_neurons+1))/nmpy.sqrt(self.hidden_neurons+1)

        # bias of gates
        self.forgt_biaass = nmpy.random.random((hidden_neurons, 1))
        self.input_bias = nmpy.random.random((hidden_neurons, 1))
        self.cell_bias = nmpy.random.random((hidden_neurons, 1))
        self.otpt_biass = nmpy.random.random((hidden_neurons, 1))
        
        # Calculating weights for the calculation of final o/p.
        self.finall_weightt = nmpy.random.random((1, hidden_neurons))

        # biass of final o/p
        self.final_output_biass= nmpy.random.random((1, 1))
    
    # Calculating Forward Progagation Function
    def forward_propagtn(self, inpt):
        inpt = nmpy.array(inpt)
        self.inpt = inpt

        for x in range(1, self.cells):
            frgt_temp = expit(self.wt_f @ nmpy.vstack((self.hid_gt_stat[x-1], self.inpt[x])) + self.forgt_biaass)
            i_temp = expit(self.weight_of_inpt@ nmpy.vstack((self.hid_gt_stat[x-1], self.inpt[x])) + self.input_bias)
            cand_temp = self.tanhh_func(self.weigt_of_c @ nmpy.vstack((self.hid_gt_stat[x-1], self.inpt[x])) + self.cell_bias)
            temp_c = frgt_temp*self.c_state[x-1] + i_temp* cand_temp
            o_temp = expit(self.weightt_otpt @ nmpy.vstack((self.hid_gt_stat[x-1], self.inpt[x])) + self.otpt_biass)
            h_temp = o_temp*self. tanhh_func(temp_c)
            
            # Storing all the result calculated gate values  to be used for the next cell state.
            self.frgt_gt_stat[x] = frgt_temp
            self.ipt_gt_stat[x] = i_temp
            self.state_of_cand_gate[x] = cand_temp
            self.c_state[x] = temp_c      
            self.o_p_gt_stat[x] = o_temp
            self.hid_gt_stat[x] = h_temp
            
        # Returning the result o/p
        y_predctns = self.finall_weightt @ self.hid_gt_stat[-1] + self.final_output_biass
        return y_predctns
    
    # This is the function to calculate backward propagation in LSTM cell.
    def backwrd_propagtn_passs(self, y_true, y_predctns):

        #These are used to  Storeee delta values  for all the gates in the LSTM
        frgt_val_delt = [nmpy.zeros((self.hidden_neurons,1)) for _ in range(self.cells+1)]
        inptt_val_delt = [nmpy.zeros((self.hidden_neurons,1)) for _ in range(self.cells+1)]
        candidt_val_delt = [nmpy.zeros((self.hidden_neurons,1)) for _ in range(self.cells+1)]
        cellVal_delt = [nmpy.zeros((self.hidden_neurons,1)) for _ in range(self.cells+1)]
        otptt_val_delt = [nmpy.zeros((self.hidden_neurons,1)) for _ in range(self.cells+1)] 
        hidenn_val_delt = [nmpy.zeros((self.hidden_neurons,1)) for _ in range(self.cells+1)]
       
        dweightt_otpt = nmpy.zeros_like(a = self.weightt_otpt)
        d_otpt_biass = nmpy.zeros_like(a = self.otpt_biass)
        
        dweight_of_inpt = nmpy.zeros_like(a = self.weight_of_inpt)
        d_input_bias = nmpy.zeros_like(a = self.input_bias)
        
        dwt_f = nmpy.zeros_like(a = self.wt_f)
        d_forgt_biaass = nmpy.zeros_like(a = self.forgt_biaass)

        d_cell_bias = nmpy.zeros_like(a = self.cell_bias)    
        dweigt_of_c = nmpy.zeros_like(a = self.weigt_of_c)
        
        d_finall_weightt = nmpy.zeros_like(a = self.finall_weightt)
        d_final_output_biass = nmpy.zeros_like(a = self.final_output_biass)  

        # these are used to Calculte the final delta values  of its bias & weight
        delt_val_e = y_true - y_predctns
        d_finall_weightt = delt_val_e * self.hid_gt_stat[-1].T
        d_final_output_biass = delt_val_e 

        for k in reversed(range(self.cells)):

            hidenn_val_delt[k] = self.finall_weightt.T @ delt_val_e + hidenn_val_delt[k+1]
            otptt_val_delt[k] = self. tanhh_func(self.c_state[k]) * hidenn_val_delt[k] * self. sigmoid_deriv_func(self.hid_gt_stat[k])
            cellVal_delt[k] = self.o_p_gt_stat[k] * hidenn_val_delt[k] * self. tanhh_func_prime(self.c_state[k]) + cellVal_delt[k+1]
            frgt_val_delt[k] = self.c_state[k-1] * cellVal_delt[k] * self. sigmoid_deriv_func(self.frgt_gt_stat[k])
            inptt_val_delt[k] = self.state_of_cand_gate[k] * cellVal_delt[k] * self. sigmoid_deriv_func(self.ipt_gt_stat[k])
            candidt_val_delt[k] = self.ipt_gt_stat[k] * cellVal_delt[k] * self. tanhh_func_prime(self.state_of_cand_gate[k])
            
            z = nmpy.vstack((self.hid_gt_stat[k-1], self.inpt[k]))
            

            dweight_of_inpt=   dweight_of_inpt+inptt_val_delt[k] @ z.T
            d_input_bias =  d_input_bias+inptt_val_delt[k]
        
            dwt_f =  dwt_f+frgt_val_delt[k] @ z.T
            d_forgt_biaass =  d_forgt_biaass+ frgt_val_delt[k]
            
          
            dweightt_otpt =  dweightt_otpt+otptt_val_delt[k] @ z.T
            d_otpt_biass = d_otpt_biass+ otptt_val_delt[k]

            
            dweigt_of_c = dweigt_of_c +cellVal_delt[k] @ z.T
            d_cell_bias = d_cell_bias+ cellVal_delt[k]            

        return d_finall_weightt, d_final_output_biass, dwt_f/self.cells, d_forgt_biaass/self.cells, dweight_of_inpt/self.cells, d_input_bias/self.cells, dweightt_otpt/self.cells, d_otpt_biass/self.cells, dweigt_of_c/self.cells, d_cell_bias/self.cells

   
    def fit(self, epochs, x, y_true, x_valid=None, y_true_valid=None):

        training_loss_list, validation_loss_list = [], []
        
        for epoch in range(epochs):
            print("epoch no:",epoch)
            trainng_loss = 0
            valdtn_of_loss = 0
            for i in range(len(x)):
                y_predctns = self.forward_propagtn(x[i])

                d_finall_weightt, d_final_output_biass, dwt_f, d_forgt_biaass, dweight_of_inpt, d_input_bias, dweightt_otpt, d_otpt_biass, dweigt_of_c, d_cell_bias = self. backwrd_propagtn_passs(y_true[i], y_predctns)

                self.wt_f = self.wt_f + self.lr_rate * dwt_f
                self.forgt_biaass = self.forgt_biaass + self.lr_rate * d_forgt_biaass

                self.weight_of_inpt= self.weight_of_inpt+ self.lr_rate * dweight_of_inpt
                self.input_bias = self.input_bias + self.lr_rate * d_input_bias

                self.weigt_of_c= self.weigt_of_c + self.lr_rate * dweigt_of_c
                self.cell_bias = self.cell_bias + self.lr_rate * d_cell_bias

                self.weightt_otpt = self.weightt_otpt + self.lr_rate * dweightt_otpt
                self.otpt_biass = self.otpt_biass + self.lr_rate * d_otpt_biass                

                self.finall_weightt = self.finall_weightt + self.lr_rate * d_finall_weightt
                self.final_output_biass = self.final_output_biass + self.lr_rate * d_final_output_biass 

                trainng_loss += ((y_true[i] - y_predctns)**2)/2

            training_loss_list.append(trainng_loss)

            if x_valid is not None and y_true_valid is not None:
                y_predictn_validtn = self.predict_stock_value(x_valid)
                y_predictn_validtn = y_predictn_validtn.reshape((y_predictn_validtn.shape[0], 1))
                y_true_valid = y_true_valid.reshape((y_true_valid.shape[0], 1))

                valdtn_of_loss = nmpy.sum( (y_true_valid - y_predictn_validtn)**2 , axis =0)/2
                validation_loss_list.append( valdtn_of_loss)

        if x_valid is not None and y_true_valid is not None:
            return nmpy.concatenate(training_loss_list), nmpy.concatenate(validation_loss_list)
    
    # This is the function to predict the stock market value for the following day
    def predict_stock_value(self, x):         
        y_prediction = []
        l = len(x)
        for i in range(l):
            y_prediction.append(self.forward_propagtn(x[i]) )            
        return nmpy.concatenate(y_prediction)
    
#below are tanh,sigmoid, derivate of tanh functions.
    def tanhh_func(self, x):
        tanhh_val = nmpy.tanh(x)
        return tanhh_val

    def sigmoid_deriv_func(self, x):
        return expit(x) * (1 - expit(x))

    def tanhh_func_prime(self, x):
        tan_prime = 1-(x**2)
        return tan_prime

snp_dataset = pd.DataFrame(data['close']).reset_index()['close']
def normalize(val_list):
    minimumm_val = min(val_list)
    maxmum_val = max(val_list)
    range_val = maxmum_val - minimumm_val
    normalized_list = []
    for val in val_list:
        norm_val = (val-minimumm_val)/range_val
        normalized_list.append(norm_val)
    return normalized_list

normalized_snp_dataset = pd.Series(normalize(snp_dataset))
def generate_val_partitions(data, input_size, output_size):
    input_list, output_list = [], []
    for i in data.index:
        output_index = i + input_size - 1 
        if output_index + output_size > data.index[-1]:
            break            
        x_list, y_list = data.loc[i:output_index], data[output_index + output_size]
        input_list.append(x_list)
        output_list.append(y_list)
    return nmpy.array(input_list), nmpy.array(output_list)
    
train_vals, test_vals = train_test_split(normalized_snp_dataset, test_size=0.2, shuffle=False)
input_size = 10
output_size = 1

X_train, y_train = generate_val_partitions(train_vals, input_size, output_size)
X_test, y_test = generate_val_partitions(test_vals, input_size, output_size)

dbutils.widgets.text("hidden_neurons", "15") # 15
hidden_neurons = int(dbutils.widgets.get("hidden_neurons"))

dbutils.widgets.text("epochs", "400") # 400
epochs = int(dbutils.widgets.get("epochs"))
dbutils.widgets.text("learning_rate", "0.1") # 0.1
learning_rate = float(dbutils.widgets.get("learning_rate"))
lstm = LSTM(hidden_neurons,input_size,learning_rate)
lstm.fit(epochs, X_train, y_train)

train_predictions = lstm.predict_stock_value(X_train)
test_predictions = lstm.predict_stock_value(X_test)

plt.plot(train_predictions)
plt.plot(y_train)
plt.legend(['Training Predictions', 'Training Observations'])
plt.show()

plt.plot(test_predictions)
plt.plot(y_test)
plt.legend(['Testing Predictions', 'Testing Observations'])
plt.show()

# Calculate the RMSE value and MAPE value for train and test sets
print("RMSE for training set: ", mse(y_train, train_predictions, squared = False))
print("RMSE for testing set:  ", mse(y_test, test_predictions, squared = False))
print("MAE for training set: ", mae(y_train, train_predictions))
print("MAE for testing set:  ", mae(y_test, test_predictions))
