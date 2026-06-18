import numpy as np

class KalmanFilter():
    def __init__(self,time=1/30): # 30 frames per second

        self.time = time
        
        self.x = np.zeros((4, 1))  # x, y, vx, vy 

        self.P = np.diag([100, 100, 100, 100]) # x, y, vx, vy, high number emans they are unknown accuracy of the system

        self.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])

        self.TransitionModel = np.array([[1, 0, self.time, 0],
                                        [0, 1, 0, self.time],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])

        self.G = np.array([[0.5*self.time**2, 0],
                        [0, 0.5*self.time**2],
                        [self.time, 0],
                        [0, self.time]])
        
        # Process noise
        sigma = 12         #2.315  # was 12 changes to 4 to test centre prediction
        # Controls how much you trust the motion model vs measurements
        #Higher = filter reacts faster to changes, but jitterier
        #Lower = smoother but slower to respond to direction changes

        self.Q = self.G @ self.G.T * sigma**2
        
        sigmaSensor = 3
        # Controls how much you trust the camera detections
        # Higher = trusts the model more, ignores noisy detections
        # Lower = trusts detections more, follows them closely

        self.R = np.diag([sigmaSensor**2, sigmaSensor**2])

        # self.x = np.zeros((6, 1)) # X,Y,VelocityX,VelocityY,AccelerationX,AccelerationY

        # self.P = np.diag([100000, 100000, 100000, 100000, 100000, 100000  ]) # 1000 in diagnals to say that we dont know where we are, and sensor reading likely wrong, usually sigma**2

        # self.H = np.array([[1, 0, 0, 0, 0, 0],
        #                    [0, 1, 0, 0, 0, 0]]) #extracts positions X and Y (measured variable)
        
        # self.time = time

        # #time = 1/(150/4)
        
        # self.TransitionModel = np.array([[1,    0, self.time,    0,    0.5*self.time**2,    0], # X position
        #                                  [0,    1,    0, self.time,    0,    0.5*self.time**2], # Y position
        #                                  [0,    0,    1,    0,    self.time,    0], # X velocity
        #                                  [0,    0,    0,    1,    0,    self.time], # Y velocity
        #                                  [0,    0,    0,    0,    1,       0], # X Acceleration
        #                                  [0,    0,    0,    0,    0,       1]])# Y Acceleration 
         
        # self.G = np.array([
        # [0.5*self.time**2, 0],  # X Position
        # [0,0.5*self.time**2],  # Y Position
        # [self.time,0],  # X Velocity
        # [0,self.time],  # Y Velocity
        # [1,0],     # X Acceleration
        # [0,1]])    # Y Acceleration
        # #50 = 2.65
        # sigma = 5  #Higher sigma = faster response, Too high causes jitters
        # self.Q = self.G @ self.G.T * sigma**2 #Noise in 2D for X and Y varialbes
        # #self.Q = np.diag([2, 2, 25, 25, 25, 25]) #Q  10 for noisy velocity because it can change frequently

        # sigmaSensor = 25**2
        # self.R = self.R = np.diag([sigmaSensor,sigmaSensor]) #Measurement noise  R = sigma **2   25 = 5 pixels, 100 = 10 pixels
    
    def Predict(self):
        self.x = (self.TransitionModel @ self.x) #plug in x and velocity x to motion model  (B * u not included(no control input))
        
        self.P = (self.TransitionModel @ self.P @ self.TransitionModel.T) + self.Q # P = what sensor to trust in diagnol R -> L, How x and y correlate in diagnol L -> R
        return self.x.copy()
    
    def Update(self,z):
        z = np.array([z]).reshape((2,1))
        
        y = z - (self.H @ self.x) #Difference between sensor and model
        S = self.H @ self.P @ self.H.T + self.R #Innovation (residual) covariance ?????? need to add + R = noise
        K = self.P @ self.H.T @ np.linalg.inv(S) #Kalman Gain
        
        self.x = self.x + (K @ y) #Update state estimate with observation
        I = np.eye(self.P.shape[0])
        self.P = (I - (K @ self.H)) @ self.P #Update covariance estimate Pk|k-1


    def predictFrame(self, steps):
        future = self.x.copy()
        for _ in range(steps):
            future = self.TransitionModel @ future
        return future
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_features, History_length, Prediction_length, Output_dim):
        super(LSTM, self).__init__()
        #input_dim = len(x_train[0])
        input_dim = input_features       # vx and vy as features
        output_dim = Output_dim*Prediction_length     # Predicting 10 frames ahead, each with vx and vy
        num_layers= 3
        hidden_dim= 64*2
        droplayer= 0.1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=droplayer
        )
        
        # Final NN to map the hidden state to 10-frame prediction
        self.head = nn.Sequential(
            nn.LayerNorm(  hidden_dim + input_dim),  # Normalize combined input
            nn.Linear(hidden_dim + input_dim, 128*2),
            nn.LeakyReLU(),
            nn.Dropout(droplayer),
            nn.Linear(128*2, 128*4),
            nn.LeakyReLU(),
            nn.Dropout(droplayer),
            nn.Linear(128*4, 128*3),
            nn.LeakyReLU(),
            nn.Dropout(droplayer),
            nn.Linear(128*3, output_dim)
        )

    def forward(self, x):
        #res = self.residual_block(x[:, -1, :])  # Residual connection from the last input time step

        #lstm_out : (batch size, sequence length, hidden dim)
        #lstm_out contains all hidden state values for every time step
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        #LAST hidden state
        last_time_step = lstm_out[:, -1, :] 

        #concatenate the residual block/input block and lstm output
        combined = torch.cat([last_time_step, x[:, -1, :]], dim=1)
        
        #Predict 20 values 10 frames * vx, vy
        out = self.head(combined)  # Use combined input to head
        return out
    


class LSTM_Sim():
    def __init__(self):
        #Run on GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LSTM_Sim running on: {self.device}")

        # ── Init model I/O ───
        data = torch.load("models/TrainData_train.pt", map_location=self.device,weights_only=False)
        self.input_features = data['Features']
        self.History_length = data['History_length']
        self.Prediction_length = data['Prediction_length']
        self.Output_length = data['Output']

        # ── build and load model ───
        self.model = LSTM(self.input_features, self.History_length, self.Prediction_length, self.Output_length).to(self.device)
        self.model.load_state_dict(torch.load("models/LSTM.pth", map_location=self.device))
        self.model.eval()

        # ── load normalisation stats ───
        norm = torch.load("models/LSTMNorm.pt", map_location=self.device)
        self.x_mean = norm["x_mean"].clone().detach().to(torch.float32)
        self.x_std  = norm["x_std"].clone().detach().to(torch.float32)
        self.y_mean = norm["y_mean"].clone().detach().to(torch.float32)
        self.y_std  = norm["y_std"].clone().detach().to(torch.float32)

    def predict_point(self,features):
        
        # Convert into tensor and add a dimension to match size
        features_tensor = torch.from_numpy(features).float().to(self.device)  # (1, H, 7)
        # Normalize
        features_norm = (features_tensor - self.x_mean.to(self.device)) / self.x_std.to(self.device)

        with torch.no_grad():
            output = self.model(features_norm)

        # Denormalize
        output = output.view(-1, self.Prediction_length, self.Output_length)
        output = output * self.y_std.to(self.device) + self.y_mean.to(self.device)

        return output.cpu().numpy()
    





class Attention(nn.Module):
    def __init__(self, d_k, seq_len):
        super(Attention, self).__init__()
        self.d_k = d_k
        
        #Weights for attention qkv
        self.weight_q = nn.Parameter(torch.randn((self.d_k, self.d_k)))
        self.weight_k = nn.Parameter(torch.randn((self.d_k, self.d_k)))
        self.weight_v = nn.Parameter(torch.randn((self.d_k, self.d_k)))

        self.weight_o = nn.Parameter(torch.randn(d_k, d_k))
        
        #Weight for Time
        self.recency_bias = nn.Parameter(torch.zeros((seq_len, seq_len)))

        #Drop out
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, x):
        seq_len = x.shape[1]

        Q = x @ self.weight_q       # query weight  (batch, history, x & y) @ (10 , 2)
        K = x @ self.weight_k       # key weight
        V = x @ self.weight_v       # value weight

        K_trans = K.transpose(-2, -1) # Dont transpose the Batch

        QK = (Q @ K_trans) / K.shape[-1] ** 0.5  # Dot product then divide for stability

        # Casual Attention - So the start point does not see future points
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        QK = QK.masked_fill(causal_mask, float('-inf'))  # block future frames

        QK = QK + self.recency_bias

        weights = torch.softmax(QK, dim=-1)

        weights = self.attn_dropout(weights)

        attended = weights @ V

        output = attended @ self.weight_o

        return output + x # Residual to maintain the values + attention


# class Transformer(nn.Module):
#     def __init__(self, input_features, History_length, Prediction_length, Output_dim):
#         super(Transformer, self).__init__()

#         droplayer= 0.2

#         # Position Embedding, needed becasue attention cant tell time
#         self.pos_embedding = nn.Embedding(History_length, input_features)

#         self.History_length = History_length

#         self.d_k = input_features//2 # Divide by 2 for multihead attention
        
#         # Seperate Attention layer for velocity and acceleration
#         self.att_vel = Attention(self.d_k,History_length) 
#         self.att_acc = Attention(self.d_k,History_length)

#         self.att_vel2 = Attention(self.d_k,History_length) 
#         self.att_acc2 = Attention(self.d_k,History_length)

#         output_dim = Output_dim * Prediction_length
        
#         # Final NN to map the hidden state to 10-frame prediction
#         head_input = History_length * (input_features + 8)
#         self.head = nn.Sequential(
#             nn.LayerNorm(head_input),     # 80
#             nn.Linear(head_input, 128),
#             nn.GELU(),
#             nn.Dropout(droplayer),
#             nn.Linear(128, 64),
#             nn.GELU(),
#             nn.Linear(64, output_dim)     # 20 # Predicting 10 frames ahead, each with vx and vy
#         )

#         self.MLP = nn.Sequential(
#             nn.LayerNorm(input_features),
#             nn.Linear(input_features, 32),
#             nn.GELU(),
#             nn.Dropout(droplayer),
#             nn.Linear(32, 16),
#             nn.GELU(),
#             nn.Linear(16, 8)
#         )

#         # Multi - Head Attention after MLP
#         self.att_MLP = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True, dropout=droplayer)

#         #Norm
#         self.norm1 = nn.LayerNorm(8)

   
#     def forward(self, x):

#         #Position Encoding [1-10]
#         positions = torch.arange(self.History_length, device=x.device)
#         pos_emb = self.pos_embedding(positions)
#         x = x + pos_emb.unsqueeze(0)

#         vel = x[:,:,:2]
#         acc = x[:,:,2:]
        
#         # multi-head attention layers
#         att_v = self.att_vel(vel)
#         att_a = self.att_acc(acc)
        
#         att_v2 = self.att_vel2(att_v)
#         att_a2 = self.att_acc2(att_a)

#         att_total = torch.cat([att_v2,att_a2], dim=2)

#         mlp = self.MLP(att_total)
        
#         seq_len = mlp.size(1)
#         causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=mlp.device)
#         mlp_att, _ = self.att_MLP(mlp,mlp,mlp,
#                                 attn_mask=causal_mask, is_causal=True)
#         mlp_norm = self.norm1(mlp + mlp_att)      # Residual + norm

#         #concatenate the residual input and mlp
#         combined = torch.cat([mlp_norm, x], dim=2)
#         combined = combined.reshape(x.shape[0], -1)      
        
#         #print("combined:\n",combined.shape)
#         #Predict 20 values 10 frames * vx, vy
#         out = self.head(combined)  # Use combined input to head
#         return out

import sys
sys.path.append(r"C:\Users\annam\OneDrive - South East Technological University\!Year 4\!Project\Object Prediction\ML Models")

from Transformer_Train_15_04_2026 import Transformer


class Transformer_Sim():
    def __init__(self):
        #Run on GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Transformer running on: {self.device}")

        # Init model I/O 
        data = torch.load("models/metadata.pt", weights_only=False)
        self.input_features = data['Features']
        self.History_length = data['History_length']
        self.Prediction_length = data['Prediction_length']
        self.Output_length = data['Output']

        # build and load model 
        self.model = Transformer(self.input_features, self.History_length, self.Prediction_length, self.Output_length).to(self.device)

        state_dict = torch.load("models/Transformer_xvel.pth", map_location=self.device, weights_only=True)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # remove '_orig_mod.'
            else:
                new_key = key
            new_state_dict[new_key] = value

        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=True)
        if missing:
            print(f"Missing keys: {missing[:5]}...")  # show first 5
        if unexpected:
            print(f"Unexpected keys: {unexpected[:5]}...")


        self.model.eval()

        # load normalisation stats 
        norm = torch.load("models/TransformerNorm.pt", map_location=self.device)
        self.x_mean = norm["x_mean"].clone().detach().to(torch.float32)
        self.x_std  = norm["x_std"].clone().detach().to(torch.float32)
        self.y_mean = norm["y_mean"].clone().detach().to(torch.float32)
        self.y_std  = norm["y_std"].clone().detach().to(torch.float32)

    def predict_point(self,features):
        
        # Convert into tensor and add a dimension to match size
        features_tensor = torch.from_numpy(features).float().to(self.device)  # (1, H, 7)
        # Normalize
        features_norm = (features_tensor - self.x_mean.to(self.device)) / self.x_std.to(self.device)
        #features_norm = torch.clamp(features_norm, -3.0, 3.0) # was getting issue with positon on edge being 4 std away which cause issues
                

        with torch.no_grad():
            output = self.model(features_norm)

        # Denormalize
        output = output.view(-1, self.Prediction_length, self.Output_length)
        output = output * self.y_std.to(self.device) + self.y_mean.to(self.device)

        return output.cpu().numpy()


    





