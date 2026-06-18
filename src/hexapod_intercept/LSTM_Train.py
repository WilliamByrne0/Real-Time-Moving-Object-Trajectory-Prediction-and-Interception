import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Force cuDNN optimized kernels
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # auto-picks fastest kernel

# Use TensorFloat32 for faster matmul on Ampere GPUs (RTX 30xx+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

scaler = torch.amp.GradScaler(device='cuda')

VELOCITY_GAIN = 10
POSITION_GAIN = 1


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
    
if __name__ == "__main__":
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load Train/Test split
    train_data        = torch.load("models/TrainData_train.pt", weights_only=False)
    History_length    = train_data['History_length']
    Prediction_length = train_data['Prediction_length']
    input_features    = train_data['Features']
    Output_dim        = train_data['Output']
    train_pairs       = train_data['training_pairs']

    test_data  = torch.load("models/TrainData_test.pt", weights_only=False)
    test_pairs = test_data['training_pairs']

    # Split data into prediction, position and current position for training
    x_train   = torch.tensor([inp  for _, _,   inp in train_pairs], dtype=torch.float32)
    y_train   = torch.tensor([pred for _, pred, _  in train_pairs], dtype=torch.float32)

    x_test   = torch.tensor([inp  for _, _,   inp in test_pairs], dtype=torch.float32)
    y_test   = torch.tensor([pred for _, pred, _  in test_pairs], dtype=torch.float32)
    

    pos_future_train = torch.tensor([pos[0] for pos, _, _ in train_pairs], dtype=torch.float32)  # (N, 30, 2)
    pos_cur_train    = torch.tensor([pos[1] for pos, _, _ in train_pairs], dtype=torch.float32)  # (N, 2)

    pos_future_test  = torch.tensor([pos[0] for pos, _, _ in test_pairs],  dtype=torch.float32)
    pos_cur_test     = torch.tensor([pos[1] for pos, _, _ in test_pairs],  dtype=torch.float32)


    x_train   = x_train.view(-1, History_length, input_features)        # (num_samples, History_length, num_features)
    y_train   = y_train.view(-1, Prediction_length, Output_dim)         # (num_samples, Prediction_length, num_features)
    #pos_train = pos_train.view(-1, 2, Prediction_length, 2)             # N frame ahead positions and Actual positions

    x_test   = x_test.view(-1, History_length, input_features)
    y_test   = y_test.view(-1, Prediction_length, Output_dim)
    #pos_test = pos_test.view(-1, 2, Prediction_length, 2)

    print(f"Train samples: {x_train.shape[0]} | Test samples: {x_test.shape[0]}")

    # Normalize each feature
    x_mean = x_train.mean(dim=(0, 1), keepdim=True)   # (1, 1, features)
    x_std  = x_train.std(dim=(0, 1),  keepdim=True) + 1e-8
    y_mean = y_train.mean(dim=(0, 1), keepdim=True)
    y_std  = y_train.std(dim=(0, 1),  keepdim=True) + 1e-8
 
    x_train = (x_train - x_mean) / x_std
    x_test  = (x_test  - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_test  = (y_test  - y_mean) / y_std

    x_mean = x_mean.to(device)
    x_std  = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)

    x_test        = x_test.to(device)
    y_test        = y_test.to(device)
    pos_future_test = pos_future_test.to(device)
    pos_cur_test  = pos_cur_test.to(device)

    torch.set_printoptions(threshold=float('inf'))

    #Higher batch size = Smoother Path
    #Lower = Jagged/Noisy
    train_dataset = TensorDataset(x_train, y_train, pos_future_train, pos_cur_train)
    train_loader = DataLoader(train_dataset, batch_size=512*4, shuffle=True,
                                num_workers=6, pin_memory=True,persistent_workers=True, prefetch_factor=6)
                                # prefetch_factor : number of preloaded batches

    model = LSTM(input_features, History_length, Prediction_length, Output_dim).to(device)

    # L1Loss had the best results with 9.67 error on the sim
    criterion = nn.L1Loss()
    #criterion = nn.MSELoss() # 14.15 error
    #criterion = nn.SmoothL1Loss(beta=1.0) # 20.43 error, is meant to be the best of both worlds
    #criterion = nn.HuberLoss(delta=1.0)

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Adam usually works better than SGD
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # early stopping and lr decrease
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=30,    # slightly longer first cycle than your original 20
    T_mult=2,  # doubles each restart: 30 -> 60 -> 120
    eta_min=1e-9  # don't let LR go to zero completely
)


    # Early stopping parameters
    BestTest = float('inf')
    MaxCount = 20 # Stop if no improvement for 50 epochs
    Counter = 0
    best_epoch = 0

    TrainLossList = []
    TestLossList = []

    epoch_train_loss = 0

    for epoch in range(1000):
        #Train Mode
        model.train()
        epoch_train_loss = 0
        epoch_test_loss = 0
        for batch_idx, (batch_x, batch_y, pos_future, pos_cur) in enumerate(train_loader):
            #Put on gpu/cpu
            batch_x      = batch_x.to(device)
            batch_y      = batch_y.to(device)
            pos_future   = pos_future.to(device)
            pos_cur      = pos_cur.to(device)

            optimizer.zero_grad(set_to_none=True)# Clear old gradients first
            
            with torch.autocast(device_type='cuda'):
                #Forward pass
                predictions = model(batch_x)
                predictions = predictions.view(batch_x.shape[0], Prediction_length, Output_dim) # Reshape to (batch, Predicted Velocity, X and Y)
                
                #Denormalize
                predictions_denorm = predictions * y_std + y_mean
                batch_y_denorm = batch_y * y_std + y_mean
                PredictPositions   = pos_cur.unsqueeze(1) + predictions_denorm.cumsum(dim=1)
                
                #Loss - takes postion and velocities as the error
                # velocity_loss = criterion(predictions, batch_y)*VELOCITY_GAIN
                # position_loss = criterion(PredictPositions, pos_future)*POSITION_GAIN
                

                velocity_loss = criterion(predictions_denorm, batch_y_denorm) * VELOCITY_GAIN
                position_loss = criterion(PredictPositions, pos_future) * POSITION_GAIN
                TrainLoss = velocity_loss + position_loss

            

            #Backward pass
            scaler.scale(TrainLoss).backward()
            scaler.unscale_(optimizer) # put gradients back to original scale
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping, grad norm looks at all gradients and if sum > 1 then reduce all gradients
            scaler.step(optimizer)
            scaler.update()

            scheduler.step(epoch + batch_idx / len(train_loader))
            
            epoch_train_loss += TrainLoss.item()
            #Average Loss per batch
        epoch_train_loss /= len(train_loader)
        TrainLossList.append(epoch_train_loss)


    # ---Eval Data---
        
        if epoch % 5 == 0:
            model.eval() # Set model to evaluation mode
            with torch.no_grad():                                               # Sets the gradient calculation off
                predictions = model(x_test)                                     # Get Predictions
                predictions = predictions.view(-1, Prediction_length, 2)        # Reshape to (batch, Predicted Velocity, X and Y)

                #Denormalize
                predictions_denorm = predictions * y_std + y_mean
                y_denorm = y_test * y_std + y_mean
                PredictPositions = pos_cur_test.unsqueeze(1) + predictions_denorm.cumsum(dim=1) # Sum all predicted velocities to get positions

                #Loss - takes postion and velocities as the error
                # velocity_loss = criterion(predictions, y_test)*VELOCITY_GAIN
                # position_loss = criterion(PredictPositions, pos_future_test)*POSITION_GAIN
                # TestLoss = (velocity_loss + position_loss).item() # positionFactor * position_loss

                velocity_loss = criterion(predictions_denorm, y_denorm) * VELOCITY_GAIN
                position_loss = criterion(PredictPositions, pos_future_test) * POSITION_GAIN
                TestLoss = (velocity_loss + position_loss).item() # positionFactor * position_loss
                TestLossList.append(TestLoss)   
        
            
            

            if TestLoss < BestTest:
                BestTest = TestLoss
                Counter = 0
                best_epoch = epoch
                # Save the best model
                torch.save(model.state_dict(), 'models/LSTM.pth')
            
            else:
                Counter += 1
        
       
                print(f"Epoch {epoch} | Train: {epoch_train_loss:.4f} | Test: {TestLoss:.4f}")
                print(f"Velocity Loss: {velocity_loss:.4f} Position Loss {position_loss:.4f}\n")

        
        if Counter >= MaxCount:
            print(f"\nStopping at epoch: {epoch}")
            print(f"Best model was at epoch {best_epoch} with test loss: {BestTest:.4f}")
            break
    
    #save normalization parameters for inference
    # torch.save({
    #     "x_mean": x_mean,
    #     "x_std": x_std,
    #     "y_mean": y_mean,
    #     "y_std": y_std}, 'models/LSTMnorm.pt')
    
    torch.save({
        "x_mean": x_mean,
        "x_std":  x_std,
        "y_mean": y_mean,
        "y_std":  y_std,
    }, 'models/LSTMnorm.pt')
    


    import matplotlib.pyplot as plt
    plt.plot(TrainLossList, label="Train Loss")
    plt.plot(TestLossList, label="Test Loss")

    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.yscale('log')

    plt.text(len(TrainLossList)*0.6, max(TrainLossList)*0.5, f"Best Test Loss: {BestTest:.4f} at epoch {best_epoch}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Model")
    plt.show()