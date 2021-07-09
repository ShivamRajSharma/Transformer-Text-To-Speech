import math
import time
import torch 
import torch.nn as nn
import numpy as np


def receptive_feild_size(layers_per_stack, kernel_size):
    dilations = [2**i for i in range(layers_per_stack)]
    receptive_feild_size = (kernel_size-1)*sum(dilations) + 1
    return receptive_feild_size


def padding_calc(kernel_size, dilation):
    padding = (kernel_size - 1)*dilation
    return padding

def mixture_of_log_sampling(y, log_scale_min=-7.0, clamp_log_scale=False):
    nr_mix = y.shape[1] // 3

    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    temp = logit_probs.data.new(logit_probs.shape).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    one_hot = torch.zeros(logit_probs.shape)
    one_hot[0, -1, argmax] = 1

    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    if clamp_log_scale:
        log_scales = torch.clamp(log_scales, min=log_scale_min)

    u = means.data.new(means.shape).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x


class Conv_layers(nn.Module):
    def __init__(
        self,
        in_channels
    ):
        super(Conv_layers, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        return nn.ReLU()(self.conv(x))


class UpScalingNet(nn.Module):
    def __init__(
        self,
        hopsize,
        factor,
        cin_channels,
    ):
        super(UpScalingNet, self).__init__()
        self.hopsize = hopsize
        self.factor = factor
        self.repetitions = int(math.log(hopsize)/math.log(self.factor))
        self.forward_layers = nn.Sequential(*[
            Conv_layers(cin_channels)
            for i in range(self.repetitions)
        ])

    def forward(self, x):
        for layer in self.forward_layers:
            x = nn.functional.interpolate(x.float(), scale_factor=self.factor)
            x = layer(x)
        return x


class WaveNet(nn.Module):
    def __init__(
        self,
        layers_per_stack=20,
        stack=2,
        residual_channels=512,
        gate_channels=512,
        filter_channels=512,
        skip_out_channels=512,
        l_in_channels=-1,
        hopsize=512,
        out_channels=256,
        scaler_input=True,
        if_quantized=False,
        include_bias=True,
        loss_ = "MOL"
    ):
        super(WaveNet, self).__init__()
        self.layers_per_stack = layers_per_stack
        self.stack = stack
        self.residual_channels = residual_channels
        self.gate_channels = gate_channels
        self.skip_out_channels = skip_out_channels
        self.hopsize = hopsize
        self.scaler_input = scaler_input
        self.padding_list  = []
        self.out_channels = out_channels
        
        self.g_gate = nn.ModuleList()
        self.g_filter = nn.ModuleList()
        
        self.l_filter = nn.ModuleList()
        self.l_gate = nn.ModuleList()

        self.skip = nn.ModuleList()
        self.residual = nn.ModuleList()
        
        if self.scaler_input:
            self.conv1 = nn.Conv1d(1, residual_channels, kernel_size=1, bias=include_bias)
        else:
            self.conv1 = nn.Conv1d(out_channels, residual_channels, kernel_size=1, bias=include_bias)

        for i in range(stack):
            dilation = 1
            for layer in range(layers_per_stack):
                padding = padding_calc(2, dilation)
                self.padding_list.append(padding)
                self.g_filter.append(
                    nn.Conv1d(
                        residual_channels,
                        filter_channels,
                        kernel_size=2,
                        padding=padding,
                        dilation=dilation,
                        bias=include_bias
                    )
                )

                self.g_gate.append(
                    nn.Conv1d(
                        residual_channels,
                        gate_channels,
                        kernel_size=2,
                        padding=padding,
                        dilation=dilation,
                        bias=include_bias
                    )
                )

                self.l_gate.append(
                    nn.Conv1d(
                        l_in_channels,
                        gate_channels,
                        kernel_size=1,
                        bias=include_bias
                    )
                )

                self.l_filter.append(
                    nn.Conv1d(
                        l_in_channels,
                        filter_channels,
                        kernel_size=1,
                        bias=include_bias
                    )
                )

                self.residual.append(
                    nn.Conv1d(
                        gate_channels,
                        residual_channels,
                        kernel_size=1,
                        bias=include_bias
                    )
                )

                self.skip.append(
                    nn.Conv1d(
                        gate_channels,
                        skip_out_channels,
                        kernel_size=1,
                        bias=include_bias
                    )
                )

                dilation *= 2

        self.receptive_field = receptive_feild_size(self.layers_per_stack, kernel_size=2)
        
        self.upscaling_net = UpScalingNet(hopsize, factor=4, cin_channels=l_in_channels)
        
        self.last_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_out_channels,  skip_out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_out_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, x, local_feature, is_training=True):
        B, _, T = x.shape

        if is_training:
            local_feature = self.upscaling_net(local_feature)

        skip = 0

        x = self.conv1(x)

        for i in range(self.layers_per_stack*self.stack):
            residuals = x
            g_f = self.g_filter[i](x)[:, :, :x.shape[-1]]
            g_g = self.g_gate[i](x)[:, :, :x.shape[-1]]
            

            l_f = self.l_filter[i](local_feature)[:, :, :x.shape[-1]]
            l_g = self.l_gate[i](local_feature)[:, :, :x.shape[-1]]
            

            f, g = g_f + l_f, g_g + l_g


            x = torch.tanh(f)*torch.sigmoid(g)
            

            s = self.skip[i](x)
            

            skip += s
            

            x = self.residual[i](x)

            x = (x + residuals)*math.sqrt(0.5)

        x = self.last_conv(skip)

        return x
    

    def waveform_generation(self, g, mel_spect, device="cpu", time_scale=100, decoding="greedy"):

        mel_spect = self.upscaling_net(mel_spect)
        time_scale = max(time_scale, mel_spect.shape[-1])

        if self.scaler_input:
            global_ = torch.zeros(1, 1, 1, dtype=torch.float)

        else:
            global_ = torch.zeros(1, self.out_channels, 1, dtype=torch.float)
            
        
        output = []
        output.append(global_)

        for t in range(time_scale):
            global_features = torch.cat(output[max(t-self.receptive_field, 0): t+1], dim=-1).to(device)
            local_features = mel_spect[:, :,  max(t-self.receptive_field, 0): t+1]

            out = self.forward(global_features, local_features, is_training=False)

            if decoding == "greedy":
                out = out.permute(0, 2, 1)
                out = out.argmax(dim=-1)[:, -1]

                if self.scaler_input:
                    output.append(out)
                else:
                    x = torch.zeros(1, self.out_channels, 1)
                    x[:, out[0], :] = 1
                    out = x
                    output.append(out.detach())

            elif decoding == "greedy_autoregress":
                out = out.permute(0, 2, 1)
                out = torch.softmax(out, dim=-1)[:, -1].view(-1)
                out = np.random.choice(np.arange(256), p=out.cpu().detach().numpy())
                if self.scaler_input:
                    output.append(torch.tensor(out.detach()))
                else:
                    x = torch.zeros(1, self.out_channels, 1)
                    x[:, out, :] = 1
                    out = x
                    output.append(out.detach())

            elif (decoding == "MOL") and (self.scaler_input):
                out = mixture_of_log_sampling(out)[:, -1].unsqueeze(0).unsqueeze(1)
                output.append(out.detach())


        return output


            
if __name__ == "__main__":
    local = torch.randn(1, 20, 114, dtype=torch.float)
    # global_ = torch.randint(0, 255, (1, 1, 29120), dtype=torch.float)
    global_ = torch.randn(1, 1, 29120, dtype=torch.float)
    device = torch.device("cpu")
    local = local.to(device)
    global_ = global_.to(device)

    wavenet = WaveNet(
        layers_per_stack=10,
        stack=3,
        residual_channels=32,
        gate_channels=32,
        filter_channels=32,
        skip_out_channels=256,
        l_in_channels=20,
        hopsize=256,
        scaler_input=False,
        out_channels=30,
        include_bias=True,
        loss_ = "MOL"
    )

    wavenet = wavenet.to(device)

    start = time.time()
    out = wavenet.waveform_generation(global_, local, device=device, decoding="greedy")
    print(f"TIME TAKEN = {time.time() - start}")
    print(out.shape)