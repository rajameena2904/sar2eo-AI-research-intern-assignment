import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False, bn=True, dropout=False):
        super(UNetBlock, self).__init__()
        if up:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not bn),
                nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
                nn.Dropout(0.5) if dropout else nn.Identity(),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not bn),
                nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNet, self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down1 = UNetBlock(64, 128)
        self.down2 = UNetBlock(128, 256)
        self.down3 = UNetBlock(256, 512)
        self.down4 = UNetBlock(512, 512)
        self.down5 = UNetBlock(512, 512)
        self.down6 = UNetBlock(512, 512)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = UNetBlock(512, 512, up=True, dropout=True)
        self.up2 = UNetBlock(1024, 512, up=True, dropout=True)
        self.up3 = UNetBlock(1024, 512, up=True, dropout=True)
        self.up4 = UNetBlock(1024, 512, up=True)
        self.up5 = UNetBlock(1024, 256, up=True)
        self.up6 = UNetBlock(512, 128, up=True)
        self.up7 = UNetBlock(256, 64, up=True)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d0 = self.initial_down(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bn = self.bottleneck(d6)
        
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d6], dim=1))
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))
        u7 = self.up7(torch.cat([u6, d1], dim=1))
        return self.final_up(torch.cat([u7, d0], dim=1))

def main():
    parser = argparse.ArgumentParser(description="Inference Engine")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet().to(device)
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Missing weight checkpoint at: {args.weights}")
        
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    target_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.png')]
    
    if not target_files:
        return

    with torch.no_grad():
        for filename in target_files:
            img_path = os.path.join(args.input_dir, filename)
            sar_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if sar_img is None:
                continue
            
            if sar_img.shape[0] != 256 or sar_img.shape[1] != 256:
                sar_img = cv2.resize(sar_img, (256, 256), interpolation=cv2.INTER_AREA)

            input_tensor = sar_img.astype(np.float32) / 255.0
            input_tensor = (input_tensor - 0.5) / 0.5
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0).to(device)

            output_tensor = model(input_tensor)

            output_tensor = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            output_tensor = ((output_tensor * 0.5) + 0.5) * 255.0
            output_tensor = np.clip(output_tensor, 0, 255).astype(np.uint8)
            
            output_bgr = cv2.cvtColor(output_tensor, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_dir, filename), output_bgr)

if __name__ == "__main__":
    main()