from pyRTC.Loop import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import threading
import argparse
import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import logging
from pyRTC.hardware.models import *

torch.set_grad_enabled(False)

class predictLoop(Loop):

    def __init__(self, conf) -> None:

        super().__init__(conf)

        self.modelPath = self.confLoop["AImodel"]
        self.halfPrecision = self.confLoop["AIhalfPrecision"]
        self.device = torch.device(self.confLoop["AIdevice"])
        self.burnIn = self.confLoop["AIburnIn"]
        self.modelLoaded = False
        self.onDevice = False
        self.normMethod = 1
        self.bufferLength = self.confLoop["AIresidualLength"]
        self.modelDType = np.float32

        

        self.residuals_mean = np.zeros((self.bufferLength,))
        self.residuals_std = np.zeros((self.bufferLength,))

        if self.halfPrecision:
            self.tensorType = torch.float16
        else:
            self.tensorType = torch.float32

        self.burnCount = None
        self.pol = None
        self.cudaGraph = None
        self.model = None
        
        self.loadModel(self.modelPath)
        self.resetBuffer()

        self.num2show = 4
        exampleIn = self.pol.detach().cpu().numpy()[:,-self.num2show:,...]
        exampleOut = self.cudaGraph(self.pol).detach().cpu().numpy()
        self.modelIn =  ImageSHM("modelIn", (exampleIn.size,), self.modelDType)
        self.modelOut =  ImageSHM("modelOut", (exampleOut.size,), self.modelDType)
        

        return
    
    # @profile_function
    def loadModel(self, pathToModel):
        
        nb = 4
        nf = 64
        try:
            self.model = DenseNet(block_num=nb, num_features=nf)
            self.model.load_state_dict(torch.load(f'/home/whetstone/Downloads/torch/torch/logs/{pathToModel}/netG_epoch_74.pth'))

            self.model = self.model.to(self.device).eval()
            self.x = torch.randn(1, 20, 48, 24, device=self.device)

            if self.halfPrecision:
                self.model = self.model.half().eval()
                self.x = self.x.half()
                # self.gCM = self.gCM.half()

                #     # hidet.torch.dynamo_config.correctness_report()
                #     self.cudaGraph = torch.compile(model, backend='hidet')
                #     self.cudaGraph(self.x)

            self.cudaGraph = torch.compile(self.model, mode='max-autotune-no-cudagraphs').eval() #, mode='max-autotune')

            output = self.cudaGraph(self.x)

            print(f'Model Loaded {output.shape}')
        
        except Exception as ex:
            logging.exception('Error while loading the pytorch model')

    # @profile_function
    def resetBuffer(self):
        self.pol = torch.zeros_like(self.x) #torch.randn(1, 20, 48, 24, device=self.device, dtype=self.x.dtype)
        self.burnCount = -1
        self.residuals_mean *= 0 #np.zeros((self.confLoop["AIresidualLength"]))
        self.residuals_std *= 0 #np.zeros((self.confLoop["AIresidualLength"]))

        return

    # Put relevant variables on GPU
    def toDevice(self):

        # IM
        if not torch.is_tensor(self.IM.dtype):
            self.IM = torch.from_numpy(self.IM)

        self.fIM = torch.from_numpy(self.fIM).to(self.device)
        self.IM = self.IM.to(self.device)
        self.gIM = self.gain * self.IM

        # gCM
        if not torch.is_tensor(self.gCM.dtype):
            self.gCM = torch.from_numpy(self.gCM)

        if not torch.is_tensor(self.CM.dtype):
            self.CM = torch.from_numpy(self.CM)

        self.gCM = self.gCM.to(self.device)
        self.CM = self.CM.to(self.device)

        self.onDevice = True

    def saveResiduals(self, filename: str):

        np.save(f'{filename}_mean', self.residuals_mean)
        np.save(f'{filename}_std', self.residuals_std)

    def saveIM(self, filename=''):
        if filename == '':
            filename = self.IMFile
        np.save(filename, self.IM.detach().cpu().numpy())


    def updateCorrectionPOL(self, correction: Tensor, slopes: Tensor) -> Tensor:
            
        # Compute POL Slopes s_{POL} = s_{RES} + IM*c_{n-1}
        # print(f'slopes: {slopes.shape}, IM: {self.IM.shape}, corr: {correction.shape}')
        s_pol = slopes - torch.matmul(self.fIM, correction)

        # Update Command Vector c_n = g*CM*s_{POL} + (1 − g) c_{n-1}  https://arxiv.org/pdf/1903.12124.pdf Eq 3
        return (1-self.gain)*correction - torch.matmul(self.gCM, s_pol)

    def setNormMethod(self, n):
        self.normMethod = n
    # @profile_function
    def standardIntegratorPOL(self):

        self.burnCount += 1
        currentCorrection = torch.from_numpy(self.wfcShm.read()).to(self.device)
        slopes = torch.from_numpy(self.wfsShm.read()).to(self.device)

        # next_pol = np.zeros((self.pol.shape[-2], self.pol.shape[-1])) #self.wfsShm.read()
        
        next_pol = self.updateCorrectionPOL(currentCorrection, slopes)
            
        # norm_max = (next_pol-norm_min).std()

        # Shift & Add data to tensor
        self.pol = torch.roll(self.pol, -1, 1)
        self.pol[:, -1, ...] = torch.matmul(self.IM, next_pol).reshape(1, 1, 48, 24).type(self.tensorType)
        
        if self.normMethod == 0:
            norm_mean = self.pol.mean()
            norm_std = self.pol.std()
        elif self.normMethod == 1:
            norm_mean = next_pol.mean()
            norm_std = next_pol.std()
        elif self.normMethod == 2:
            norm_mean = self.pol.mean()
            norm_std = 1.
        elif self.normMethod == 3:
            norm_mean = next_pol.mean()
            norm_std = 1.
        else:
            norm_mean = 0.0632
            norm_std = 0.9381
        
        #Data to send to network
        modelIn = (self.pol-norm_mean)/norm_std
        #Send the Shm for debug
        self.modelIn.write(modelIn[:,-self.num2show:,...].detach().cpu().numpy().flatten().astype(self.modelDType))
        #Send to network
        net_output = -20.*self.cudaGraph(modelIn).flatten().type(torch.float32)
        #Send output to shm for debug
        self.modelOut.write(net_output.detach().cpu().numpy().flatten().astype(self.modelDType))
        #Turn into new correction for DM
        newCorrection = torch.matmul(self.CM, (net_output*norm_std) + norm_mean)
        
        if self.burnCount > self.burnIn:
            newCorrection = newCorrection.detach().cpu().numpy().astype(self.modelDType)
        else:
            newCorrection = next_pol.detach().cpu().numpy().astype(self.modelDType) #currentCorrection - torch.matmul(self.gCM, next_pol).detach().cpu().numpy().astype(np.float32)
        
        newCorrection[self.numActiveModes:] = 0
        
        if self.burnCount > self.burnIn-5:
            np.savez(f"j12moving_predict_debug_info_{self.burnCount}", 
                    currentCorrection=currentCorrection.detach().cpu().numpy(), 
                    slopes=slopes.detach().cpu().numpy(), 
                    next_pol=next_pol.detach().cpu().numpy(),
                    self_pol=self.pol.detach().cpu().numpy(),
                    norm_mean=norm_mean.detach().cpu().numpy(), 
                    norm_std=norm_std.detach().cpu().numpy(), 
                    model_in=self.modelIn.read_noblock_safe(),
                    model_out=self.modelOut.read_noblock_safe(),
                    #modelIn=modelIn.detach().cpu().numpy(), 
                    net_output=net_output.detach().cpu().numpy(), 
                    newCorrection=newCorrection)
            
            if self.burnCount > self.burnIn + 5:
                assert(False)

        if self.burnCount < self.bufferLength:
            self.residuals_mean[self.burnCount] = currentCorrection.detach().cpu().numpy().mean()
            self.residuals_std[self.burnCount] = currentCorrection.detach().cpu().numpy().std()

        self.wfcShm.write(newCorrection)

        return
    
    def standardIntegrator(self):
        raise NotImplementedError('Model Requires POL Integration')


if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    os.sched_setaffinity(pid, {conf["loop"]["affinity"],})
    decrease_nice(pid)

    loop = predictLoop(conf=conf)
    
    l = Listener(loop, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)