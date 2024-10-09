import numpy as np
import torch
import torch.nn as nn
from pyRTC.Loop import *
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import logging
import matplotlib
import matplotlib.pyplot as plt
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('plt').setLevel(logging.WARNING)

class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        N, M = image_size  # Image dimensions

        # Define the convolutional layers with smaller kernel sizes and strides
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Output: (batch, 64, N/2, M/2)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (batch, 128, N/4, M/4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: (batch, 256, N/8, M/8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Apply Adaptive Average Pooling to reduce spatial dimensions to (1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Fully connected layer to output the probability
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, 1, N, M)
        out = self.model(x)
        out = self.fc(out)
        return out


class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = tuple(k // 2 for k in self.kernel_size)
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and previous hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (batch, input_channels + hidden_channels, height, width)
        
        # Compute gates
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        
        # Input gate
        i = torch.sigmoid(cc_i)
        # Forget gate
        f = torch.sigmoid(cc_f)
        # Output gate
        o = torch.sigmoid(cc_o)
        # Cell gate
        g = torch.tanh(cc_g)
        
        # New cell state
        c_next = f * c_cur + i * g
        # New hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width, device=None):
        device = device or self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM module.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()
        
        # Ensure hidden_channels and kernel_size are lists
        hidden_channels = self._to_list(hidden_channels, num_layers)
        kernel_size = self._to_list(kernel_size, num_layers)
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        
        # Create a list of ConvLSTMCells
        cells = []
        for i in range(num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels[i - 1]
            cells.append(ConvLSTMCell(
                input_channels=cur_input_channels,
                hidden_channels=self.hidden_channels[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))
        self.cells = nn.ModuleList(cells)
    
    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters:
        - input_tensor: (batch, seq_len, channels, height, width)
        - hidden_state: list of tuples (h, c) for each layer
        """
        if not self.batch_first:
            # Convert to (batch, seq_len, channels, height, width)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, height=h, width=w, device=input_tensor.device)
        
        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cells[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                             cur_state=[h, c])
                output_inner.append(h)
            
            # Stack outputs for each time step
            layer_output = torch.stack(output_inner, dim=1)  # (batch, seq_len, hidden_channels, height, width)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        # Return outputs and last states
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, height, width, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cells[i].init_hidden(batch_size, height, width, device))
        return init_states
    
    @staticmethod
    def _to_list(param, num_layers):
        if isinstance(param, list):
            assert len(param) == num_layers, "List length must match number of layers"
            return param
        else:
            return [param] * num_layers



class ConvLSTMModel(nn.Module):
    def __init__(self, image_size, hidden_channels, num_layers=1, kernel_size=(3, 3)):
        super(ConvLSTMModel, self).__init__()
        
        N, M = image_size  # Image dimensions
        
        self.N = N
        self.M = M
        self.hidden_channels = hidden_channels if isinstance(hidden_channels, list) else [hidden_channels] * num_layers
        
        # Define ConvLSTM
        self.convlstm = ConvLSTM(
            input_channels=1,  # Grayscale images
            hidden_channels=self.hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True
        )
        
        # Final convolution to map hidden state to output image
        self.conv_out = nn.Conv2d(
            in_channels=self.hidden_channels[-1],
            out_channels=1,  # Output is a grayscale image
            kernel_size=1
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, N, M)
        batch_size, seq_len, N, M = x.size()
        
        # Ensure input size matches model initialization
        assert N == self.N and M == self.M, "Input image size must match the initialized image size"
        
        # Add channel dimension: (batch_size, seq_len, channels=1, N, M)
        x = x.unsqueeze(2)
        
        # Forward through ConvLSTM
        layer_output_list, last_state_list = self.convlstm(x)
        
        # Get output from the last layer
        output = layer_output_list[-1]  # (batch_size, seq_len, hidden_channels[-1], N, M)
        
        # Take output at the last time step
        output = output[:, -1, :, :, :]  # (batch_size, hidden_channels[-1], N, M)
        
        # Map to output image
        output = self.conv_out(output)  # (batch_size, 1, N, M)
        
        # Optionally apply activation (e.g., sigmoid) if needed
        # output = torch.sigmoid(output)
        
        return output


class basicPredictLoop(Loop):
    def __init__(self, conf) -> None:
        self.T = conf["T"]
        self.K = conf["K"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validSubApsFile = conf["validSubApsFile"]
        self.validSubAps = np.load(self.validSubApsFile)
        self.numXSlopes = int(np.sum(self.validSubAps) / 2)
        self.numXSlopes2D = int(self.validSubAps.shape[1]/2)
        self.slopemask = self.validSubAps[:,:self.numXSlopes2D]
        self.history = np.zeros((self.K, *self.validSubAps.shape), dtype=np.float32)
        self.curSignal2D = np.zeros_like(self.history[0])
        self.history_GPU = torch.tensor(self.history, device = self.device).unsqueeze(0)
        self.history_idx = 0
        self.arangeK = torch.arange(self.K, device=self.device)
        self.s_pol = torch.zeros(np.sum(self.validSubAps), dtype=torch.float32, device=self.device)
        self.s_pol_pred = torch.zeros(np.sum(self.validSubAps), dtype=torch.float32, device=self.device)
        #Initialize the pyRTC super class
        super().__init__(conf)
        """
        Initializes the predictive controller.

        Parameters:
        - N (int): Size of the control vector.
        - T (int): System lag in frames to compensate.
        - K (int): History length (number of past frames to use).
        - hidden_size (int): Number of features in the hidden state of the LSTM.
        - num_layers (int): Number of recurrent layers in the LSTM.
        - learning_rate (float): Learning rate for the optimizer.
        - num_epochs (int): Number of epochs for training.
        - batch_size (int): Batch size for training.
        """

        self.hidden_size = conf["hidden_size"]
        self.num_layers = conf["num_layers"]
        self.learning_rate = conf["learning_rate"]
        self.num_epochs = conf["num_epochs"]
        self.batch_size = conf["batch_size"]
        self.recordLength = 0
        # Move the model and its parameters to the GPU
        
        
        # Define the LSTM-based predictive controller model
        model = ConvLSTMModel( 
                         hidden_channels=[self.hidden_size]*self.num_layers, 
                         image_size=self.validSubAps.shape,
                         num_layers=self.num_layers)
        discriminator = Discriminator(image_size=self.validSubAps.shape)
        self.model = model.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Define loss functions
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_loss = nn.L1Loss()  # You can also use nn.MSELoss()

        # Define optimizers for generator and self.discriminator
        self.optimizer_G = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        # Buffer to hold the most recent K control vectors for real-time prediction
        self.polShm = ImageSHM("pol", self.curSignal2D.shape, self.curSignal2D.dtype, gpuDevice=self.gpuDevice, consumer=False)
        self.slopesBuffer = None
        self.predict = False
        self.numRecords = 0
        self.recordLength = 0
        self.record = False
        self.gamma = 0

    def start(self):
        self.model.eval()
        return super().start()

    def toDevice(self):
        self.fIM_GPU = torch.tensor(self.fIM, device= self.device)
        self.curSignal2D_GPU = torch.tensor(self.curSignal2D, device= self.device)
        self.slopemask_GPU = torch.tensor(self.slopemask, device= self.device)
        self.validSubAps_GPU = torch.tensor(self.validSubAps, device= self.device)
        self.gCM_GPU = torch.tensor(self.gCM, device= self.device)
        return
    
    def computeCM(self):
        super().computeCM()
        self.toDevice()
        return 

    def listen(self, recordLength):
        #Turn on the loop
        if not self.running:
            self.start()
        self.recordLength = recordLength
        self.slopesBuffer = np.zeros((self.recordLength,  *self.validSubAps.shape)
                                      , dtype=np.float32)
        # self.slopesBuffer = torch.tensor(self.slopesBuffer, device=self.device)
        self.record = True
        self.numRecords = 0
        #Block until recording is done
        while self.record:
            time.sleep(1e-5)
        

    def train(self):
        """
        Trains the predictive controller model with adversarial training.

        This function incorporates adversarial training by introducing a self.discriminator network
        that distinguishes between real target images and images generated by the model (generator).
        The generator (self.model) aims to produce images that are indistinguishable from real images.

        Parameters:
        - self.slopesBuffer (np.ndarray): Array of control vectors with shape (num_samples, N, M).
        """
        if self.slopesBuffer is None:
            raise Exception("Must record data with listen() first")

        # Prepare the dataset
        input_sequences = []
        target_images = []
        num_samples = self.slopesBuffer.shape[0]

        for i in range(num_samples - self.K - self.T + 1):
            input_seq = self.slopesBuffer[i:i+self.K]  # Shape: (K, N, M)
            target = self.slopesBuffer[i+self.K+self.T-1]  # Target is T steps ahead
            input_sequences.append(input_seq)
            target_images.append(target)

        input_sequences = np.array(input_sequences)  # Shape: (num_samples, K, N, M)
        target_images = np.array(target_images)      # Shape: (num_samples, N, M)

        # Convert to PyTorch tensors
        input_sequences = torch.tensor(input_sequences, dtype=torch.float32, device=self.device)
        target_images = torch.tensor(target_images, dtype=torch.float32, device=self.device)

        # Create Dataset and DataLoaders
        dataset = torch.utils.data.TensorDataset(input_sequences, target_images)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)


        pbar = tqdm(range(self.num_epochs), unit='epoch')
        tLoss = []
        vLoss = []
        for epoch in pbar:
            # Training phase
            self.model.train()
            self.discriminator.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            for inputs, targets in dataloader:
                batch_size = inputs.size(0)
                # Generate labels
                valid = torch.ones((batch_size, 1), dtype=torch.float, device=self.device)
                fake = torch.zeros((batch_size, 1), dtype=torch.float, device=self.device)

                # ---------------------
                #  Train self.discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                # Generator output (fake images)
                gen_outputs = self.model(inputs)  # Shape: (batch_size, 1, N, M)

                # Real images (targets)
                real_images = targets.unsqueeze(1)  # Shape: (batch_size, 1, N, M)

                # self.discriminator predictions
                real_pred = self.discriminator(real_images)
                fake_pred = self.discriminator(gen_outputs.detach())

                # Compute self.discriminator loss
                d_real_loss = self.adversarial_loss(real_pred, valid)
                d_fake_loss = self.adversarial_loss(fake_pred, fake)
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()

                # Adversarial loss (Generator tries to fool the self.discriminator)
                fake_pred = self.discriminator(gen_outputs)
                g_adv_loss = self.adversarial_loss(fake_pred, valid)

                # Reconstruction loss (Generator tries to produce images similar to the target)
                g_recon_loss = self.reconstruction_loss(gen_outputs, real_images)

                # Total generator loss
                lambda_recon = 100  # Weight for reconstruction loss
                g_loss = g_adv_loss + lambda_recon * g_recon_loss

                g_loss.backward()
                self.optimizer_G.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            tLoss.append(avg_g_loss)

            # Validation phase
            self.model.eval()
            self.discriminator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in validation_loader:
                    batch_size = val_inputs.size(0)
                    valid = torch.ones((batch_size, 1), dtype=torch.float, device=self.device)

                    # Generate outputs
                    val_gen_outputs = self.model(val_inputs)  # Shape: (batch_size, 1, N, M)

                    # Real images (targets)
                    val_real_images = val_targets.unsqueeze(1)

                    # self.discriminator predictions on generated images
                    val_fake_pred = self.discriminator(val_gen_outputs)

                    # Generator loss
                    val_g_adv_loss = self.adversarial_loss(val_fake_pred, valid)
                    val_g_recon_loss = self.reconstruction_loss(val_gen_outputs, val_real_images)
                    val_g_loss = val_g_adv_loss + lambda_recon * val_g_recon_loss

                    val_loss += val_g_loss.item()

            avg_val_loss = val_loss / len(validation_loader)
            vLoss.append(avg_val_loss)

            if epoch % 5 == 0:
                # Visualization code (adjust as needed)
                with torch.no_grad():
                    sample_inputs, sample_targets = next(iter(validation_loader))
                    sample_gen_outputs = self.model(sample_inputs)

                    # Convert tensors to numpy arrays for visualization
                    sample_pred = sample_gen_outputs.cpu().numpy()[0][0]
                    sample_gt = sample_targets.cpu().numpy()[0]

                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 3, 1)
                    plt.imshow(sample_pred, cmap='gray')
                    plt.title('Generated Image')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(sample_gt, cmap='gray')
                    plt.title('Ground Truth')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(sample_pred - sample_gt, cmap='bwr')
                    plt.title('Difference')
                    plt.axis('off')

                    plt.show()

                    plt.plot(tLoss, color='k', label='Generator Loss')
                    plt.plot(vLoss, color='r', label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.yscale('log')
                    plt.legend()
                    plt.show()

            # Update the progress bar's description
            pbar.set_description(f'Epoch [{epoch+1}/{self.num_epochs}]')
            pbar.set_postfix({'G Loss': f'{avg_g_loss:.4f}', 'D Loss': f'{avg_d_loss:.4f}', 'Val Loss': f'{avg_val_loss:.4f}'})


    def runInference(self, history):
        """
        Predicts the control vector T steps ahead using the trained model.

        Parameters:
        - current_control_vector (np.ndarray): The current control vector of shape (N,).

        Returns:
        - predicted_vector (np.ndarray or None): The predicted control vector T steps ahead.
          Returns None if not enough data is available yet.
        """
        if isinstance(history, np.ndarray):
            history = torch.tensor(
                history,
                dtype=torch.float32, device = self.device
            ).unsqueeze(0)  # Shape: (K, 1, N)

        predicted_vector = self.model(history).squeeze().detach()
        return predicted_vector #.detach().cpu().numpy()#.flatten()

    def predictiveIntegrator(self):
        #Read Slopes
        if self.gpuDevice is not None:
            residual_slopes = self.signalShm.read(SAFE= False, RELEASE_GIL = self.RELEASE_GIL, GPU=True)
            currentCorrection = self.wfcShm.read(SAFE= False, RELEASE_GIL = self.RELEASE_GIL , GPU=True)
        else:
            residual_slopes = self.signalShm.read(SAFE= False, RELEASE_GIL = self.RELEASE_GIL, GPU=False)
            currentCorrection = self.wfcShm.read_noblock(SAFE= False, GPU=False)
            residual_slopes = torch.tensor(residual_slopes, device = self.device)
            currentCorrection = torch.tensor(currentCorrection, device = self.device)
        # print(f'slopes: {residual_slopes.shape}, IM: {self.IM.shape}, corr: {currentCorrection.shape}')
        self.s_pol = residual_slopes - self.fIM_GPU@currentCorrection

        self.curSignal2D_GPU[:,:self.numXSlopes2D][self.slopemask_GPU] = self.s_pol[:self.numXSlopes]
        self.curSignal2D_GPU[:,self.numXSlopes2D:][self.slopemask_GPU] = self.s_pol[self.numXSlopes:]
        
        # Update history buffer using circular buffer logic
        self.history_GPU[0][self.history_idx].copy_(self.curSignal2D_GPU)
        self.history_idx = (self.history_idx + 1) % self.K

        #Add pol_slopes to the buffer
        if self.record and self.numRecords < self.recordLength:
            self.slopesBuffer[self.numRecords] = self.curSignal2D_GPU.cpu().numpy()
            self.numRecords += 1
        elif self.numRecords == self.recordLength:
            self.record = False
            self.numRecords = 0

        #if we have enough in the buffer
        if self.predict:
            

            # Assemble the sequence using precomputed indices
            indices = (self.arangeK + self.history_idx) % self.K
            sequence = self.history_GPU[0].index_select(0, indices)
            sequence = sequence.unsqueeze(0)

            predictImage = self.runInference(sequence)
            self.polShm.write(predictImage)

            # Optimize slicing and indexing
            x_slopes = predictImage[:, :self.numXSlopes2D]
            y_slopes = predictImage[:, self.numXSlopes2D:]

            # Use boolean indexing directly
            self.s_pol_pred[:self.numXSlopes] = x_slopes[self.slopemask_GPU]
            self.s_pol_pred[self.numXSlopes:] = y_slopes[self.slopemask_GPU]
            
            #Forward propagate the correction using the model
            self.s_pol = (1-self.gamma)*self.s_pol + self.s_pol_pred*self.gamma
            # del predictImage
            # torch.cuda.empty_cache()
        else:
            self.polShm.write(self.curSignal2D_GPU)
        #Leak the current shape
        currentCorrection *= (1-self.leakyGain)
        newCorrection = (1-self.gain)*currentCorrection - torch.matmul(self.gCM_GPU,self.s_pol)
        
        if self.gpuDevice is None:
            newCorrection = newCorrection.cpu().numpy()
        #Send to the WFC
        self.sendToWfc(newCorrection, slopes=None)

        return

    def setGain(self, gain):
        super().setGain(gain)
        self.toDevice()
        return 

    def flatten(self):
        self.history *= 0
        self.history_GPU *= 0
        return super().flatten()
    
    def loadModels(self):
        # Load the entire model
        self.model = torch.load('./calib/model.pth')
        self.model.eval()  # Set the model to evaluation mode if necessary
        # Load the entire model
        self.discriminator = torch.load('./calib/discriminator.pth')
        self.discriminator.eval()  # Set the model to evaluation mode if necessary
    
    def saveModels(self):
        torch.save(self.model, './calib/model.pth')
        torch.save(self.discriminator, './calib/discriminator.pth')
        return
    
if __name__ == "__main__":

    launchComponent(basicPredictLoop, "loop", start = False)