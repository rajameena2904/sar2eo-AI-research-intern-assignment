import os
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.down1 = UNetBlock(features, features * 2, down=True)
        self.down2 = UNetBlock(features * 2, features * 4, down=True)
        self.down3 = UNetBlock(features * 4, features * 8, down=True)
        self.down4 = UNetBlock(features * 8, features * 8, down=True)
        self.down5 = UNetBlock(features * 8, features * 8, down=True)
        self.down6 = UNetBlock(features * 8, features * 8, down=True)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )
        
        self.up1 = UNetBlock(features * 8, features * 8, down=False, use_dropout=True)
        self.up2 = UNetBlock(features * 16, features * 8, down=False, use_dropout=True)
        self.up3 = UNetBlock(features * 16, features * 8, down=False, use_dropout=True)
        self.up4 = UNetBlock(features * 16, features * 8, down=False)
        self.up5 = UNetBlock(features * 16, features * 4, down=False)
        self.up6 = UNetBlock(features * 8, features * 2, down=False)
        self.up7 = UNetBlock(features * 4, features, down=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bn = self.bottleneck(d7)
        
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        return self.final_up(torch.cat([u7, d1], dim=1))

def main():
    parser = argparse.ArgumentParser(description="GalaxEye Evaluation Inference Deployment Contract")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to evaluation SAR directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save generated target images")
    parser.add_argument("--weights", type=str, required=True, help="Path to local generator checkpoint file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator(in_channels=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    valid_extensions = (".png", ".PNG")
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(valid_extensions)]

    print(f"Loaded weights checkpoint. Processing {len(input_files)} target patches.")

    with torch.no_grad():
        for filename in input_files:
            img_path = os.path.join(args.input_dir, filename)
            
            sar_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if sar_img is None:
                continue
                
            input_tensor = sar_img.astype(np.float32) / 255.0
            input_tensor = (input_tensor - 0.5) / 0.5
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0).to(device)

            output_tensor = model(input_tensor)

            output_tensor = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            output_tensor = ((output_tensor * 0.5) + 0.5) * 255.0
            output_tensor = np.clip(output_tensor, 0, 255).astype(np.uint8)
            
            output_bgr = cv2.cvtColor(output_tensor, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_dir, filename), output_bgr)

    print(f"Inference successfully compiled inside target folder: {args.output_dir}")

if __name__ == "__main__":
    main()
