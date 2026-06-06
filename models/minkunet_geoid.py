import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from models.resnet import ResNetBase


class MinkUNetBase_geoid(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])
        
        middle_inplanes = self.inplanes
        
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
                
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.dropout = ME.MinkowskiDropout(p=0.5)
        
        # classification branch
        self.inplanes = middle_inplanes
        
        self.convtr4p16s2_cls = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4_cls = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5_cls = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2_cls = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5_cls = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6_cls = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2_cls = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6_cls = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7_cls = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2_cls = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7_cls = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8_cls = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final_cls = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            1,
            kernel_size=1,
            bias=True,
            dimension=D)

    def forward(self, x, is_seg=True, is_cls=False, only_seg=False):
        
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        
        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_b4p16 = self.block4(out)
        
        if not is_cls or only_seg:
            # tensor_stride=8
            out = self.convtr4p16s2(out_b4p16)
            out = self.bntr4(out)
            out = self.relu(out)

            out = ME.cat(out, out_b3p8)
            out_b5p8 = self.block5(out)
            
            # tensor_stride=4
            out = self.convtr5p8s2(out_b5p8)
            out = self.bntr5(out)
            out = self.relu(out)

            out = ME.cat(out, out_b2p4)
            out_b6p4 = self.block6(out)
            
            # tensor_stride=2
            out = self.convtr6p4s2(out_b6p4)
            out = self.bntr6(out)
            out = self.relu(out)

            out = ME.cat(out, out_b1p2)
            out_b7p2 = self.block7(out)
            
            # tensor_stride=1
            out = self.convtr7p2s2(out_b7p2)
            out = self.bntr7(out)
            out = self.relu(out)

            out = ME.cat(out, out_p1)
            out = self.block8(out)
        
        if only_seg:
            return self.final(out)

        #! classification branch
        # tensor_stride=8
        out_cls = self.convtr4p16s2_cls(out_b4p16)
        out_cls = self.bntr4_cls(out_cls)
        out_cls = self.relu(out_cls)

        out_cls = ME.cat(out_cls, out_b3p8)
        out_cls = self.block5_cls(out_cls)
        
        # tensor_stride=4
        out_cls = self.convtr5p8s2_cls(out_cls)
        out_cls = self.bntr5_cls(out_cls)
        out_cls = self.relu(out_cls)

        out_cls = ME.cat(out_cls, out_b2p4)
        out_cls = self.block6_cls(out_cls)
        
        # tensor_stride=2
        out_cls = self.convtr6p4s2_cls(out_cls)
        out_cls = self.bntr6_cls(out_cls)
        out_cls = self.relu(out_cls)

        out_cls = ME.cat(out_cls, out_b1p2)
        out_cls = self.block7_cls(out_cls)
        
        # tensor_stride=1
        out_cls = self.convtr7p2s2_cls(out_cls)
        out_cls = self.bntr7_cls(out_cls)
        out_cls = self.relu(out_cls)

        out_cls = ME.cat(out_cls, out_p1)
        
        out_cls = self.block8_cls(out_cls)
        
        if is_cls:
            return self.final_cls(out_cls)
        
        if is_seg:
            return self.final(out), self.final_cls(out_cls)
        else:
            return out


class MinkUNet34_geoid(MinkUNetBase_geoid):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
