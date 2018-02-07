import torch as th
import math
import numpy as np

def th_iterproduct(*args):
    return th.from_numpy(np.indices(args).reshape((len(args),-1)).T)

def th_nearest_interp2d(input, coords):
    """
    2d nearest neighbor interpolation th.Tensor
    """
    # take clamp of coords so they're in the image bounds
    x = th.clamp(coords[:,:,0], 0, input.size(1)-1).round()
    y = th.clamp(coords[:,:,1], 0, input.size(2)-1).round()

    stride = th.LongTensor(input.stride())
    x_ix = x.mul(stride[1]).long()
    y_ix = y.mul(stride[2]).long()

    input_flat = input.view(input.size(0),-1)

    mapped_vals = input_flat.gather(1, x_ix.add(y_ix))

    return mapped_vals.view_as(input)


def th_affine2d(x, matrix, mode='bilinear', center=True):
    """
    2D Affine image transform on th.Tensor
    
    Arguments
    ---------
    x : th.Tensor of size (C, H, W)
        image tensor to be transformed
    matrix : th.Tensor of size (3, 3) or (2, 3)
        transformation matrix
    mode : string in {'nearest', 'bilinear'}
        interpolation scheme to use
    center : boolean
        whether to alter the bias of the transform 
        so the transform is applied about the center
        of the image rather than the origin
    Example
    ------- 
    >>> import torch
    >>> from torchsample.utils import *
    >>> x = th.zeros(2,1000,1000)
    >>> x[:,100:1500,100:500] = 10
    >>> matrix = th.FloatTensor([[1.,0,-50],
    ...                             [0,1.,-50]])
    >>> xn = th_affine2d(x, matrix, mode='nearest')
    >>> xb = th_affine2d(x, matrix, mode='bilinear')
    """

    if matrix.dim() == 2:
        matrix = matrix[:2,:]
        matrix = matrix.unsqueeze(0)
    elif matrix.dim() == 3:
        if matrix.size()[1:] == (3,3):
            matrix = matrix[:,:2,:]

    A_batch = matrix[:,:,:2]
    if A_batch.size(0) != x.size(0):
        A_batch = A_batch.repeat(x.size(0),1,1)
    b_batch = matrix[:,:,2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(1),x.size(2))
    coords = _coords.unsqueeze(0).repeat(x.size(0),1,1).float()

    if center:
        # shift the coordinates so center is the origin
        coords[:,:,0] = coords[:,:,0] - (x.size(1) / 2. - 0.5)
        coords[:,:,1] = coords[:,:,1] - (x.size(2) / 2. - 0.5)
    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1,2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,:,0] = new_coords[:,:,0] + (x.size(1) / 2. - 0.5)
        new_coords[:,:,1] = new_coords[:,:,1] + (x.size(2) / 2. - 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp2d(x.contiguous(), new_coords)
    elif mode == 'bilinear':
        x_transformed = th_bilinear_interp2d(x.contiguous(), new_coords)

    return x_transformed

###################### Rotate #####################################
class Rotate(object):

    def __init__(self, 
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.
        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = math.pi / 180 * self.value
        rotation_matrix = th.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
                                          [math.sin(theta), math.cos(theta), 0],
                                          [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       rotation_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]
        
        
########################### Translate  ########################### 

class Translate(object):

    def __init__(self, 
                 value, 
                 interp='bilinear',
                 lazy=False):
        """
        Arguments
        ---------
        value : float or 2-tuple of float
            if single value, both horizontal and vertical translation
            will be this value * total height/width. Thus, value should
            be a fraction of total height/width with range (-1, 1)
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        """
        if not isinstance(value, (tuple,list)):
            value = (value, value)

        if value[0] > 1 or value[0] < -1:
            raise ValueError('Translation must be between -1 and 1')
        if value[1] > 1 or value[1] < -1:
            raise ValueError('Translation must be between -1 and 1')

        self.height_range = value[0]
        self.width_range = value[1]
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        tx = self.height_range * inputs[0].size(1)
        ty = self.width_range * inputs[0].size(2)

        translation_matrix = th.FloatTensor([[1, 0, tx],
                                             [0, 1, ty],
                                             [0, 0, 1]])
        if self.lazy:
            return translation_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       translation_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]
        
########################### Shear ########################### 
class Shear(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = (math.pi * self.value) / 180
        shear_matrix = th.FloatTensor([[1, -math.sin(theta), 0],
                                        [0, math.cos(theta), 0],
                                        [0, 0, 1]])
        if self.lazy:
            return shear_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       shear_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]
        


########################### Zoom ###########################   
class Zoom(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Arguments
        ---------
        value : float
            Fractional zoom.
            =1 : no zoom
            >1 : zoom-in (value-1)%
            <1 : zoom-out (1-value)%
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy: boolean
            If true, just return transformed
        """

        if not isinstance(value, (tuple,list)):
            value = (value, value)
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        zx, zy = self.value
        zoom_matrix = th.FloatTensor([[zx, 0, 0],
                                      [0, zy, 0],
                                      [0, 0,  1]])        

        if self.lazy:
            return zoom_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       zoom_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]
    
  
############################  Compose ############################
class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        """
        Composes (chains) several transforms together into
        a single transform
        Arguments
        ---------
        transforms : a list of transforms
            transforms will be applied sequentially
        """
        self.transforms = transforms

    def __call__(self, *inputs):
        for transform in self.transforms:
            if not isinstance(inputs, (list,tuple)):
                inputs = [inputs]
            inputs = transform(*inputs)
        return inputs
    
################  To Tensor ################
class ToTensor(object):
    """
    Converts a numpy array to torch.Tensor
    """
    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = th.from_numpy(_input)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]
    
   
################  AddChannel ################
class AddChannel(object):
    """
    Adds a dummy channel to an image. 
    This will make an image of size (28, 28) to now be
    of size (1, 28, 28), for example.
    """
    def __init__(self, axis=0):
        """
        Adds a dummy channel to an image, also known as
        expanding an axis or unsqueezing a dim
        Arguments
        ---------
        axis : integer
            dimension to be expanded to singleton
        """
        self.axis = axis

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.unsqueeze(self.axis)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]
    
  

class Transpose(object):

    def __init__(self, dim1, dim2):
        """
        Swaps two dimensions of a tensor
        Arguments
        ---------
        dim1 : integer
            first dim to switch
        dim2 : integer
            second dim to switch
        """
        self.dim1 = dim1
        self.dim2 = dim2

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = th.transpose(_input, self.dim1, self.dim2)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class RangeNormalize(object):
    """
    Given min_val: (R, G, B) and max_val: (R,G,B),
    will normalize each channel of the th.*Tensor to
    the provided min and max values.
    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    where min' & max' are given values, 
    and min & max are observed min/max for each channel
    
    Arguments
    ---------
    min_range : float or integer
        Min value to which tensors will be normalized
    max_range : float or integer
        Max value to which tensors will be normalized
    fixed_min : float or integer
        Give this value if every sample has the same min (max) and 
        you know for sure what it is. For instance, if you
        have an image then you know the min value will be 0 and the
        max value will be 255. Otherwise, the min/max value will be
        calculated for each individual sample and this will decrease
        speed. Dont use this if each sample has a different min/max.
    fixed_max :float or integer
        See above
    Example:
        >>> x = th.rand(3,5,5)
        >>> rn = RangeNormalize((0,0,10),(1,1,11))
        >>> x_norm = rn(x)
    Also works with just one value for min/max:
        >>> x = th.rand(3,5,5)
        >>> rn = RangeNormalize(0,1)
        >>> x_norm = rn(x)
    """
    def __init__(self, 
                 min_val, 
                 max_val):
        """
        Normalize a tensor between a min and max value
        Arguments
        ---------
        min_val : float
            lower bound of normalized tensor
        max_val : float
            upper bound of normalized tensor
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _min_val = _input.min()
            _max_val = _input.max()
            a = (self.max_val - self.min_val) / (_max_val - _min_val)
            b = self.max_val- a * _max_val
            _input = _input.mul(a).add(b)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class StdNormalize(object):
    """
    Normalize torch tensor to have zero mean and unit std deviation
    """
    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.sub(_input.mean()).div(_input.std())
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class Slice2D(object):

    def __init__(self, axis=0, reject_zeros=False):
        """
        Take a random 2D slice from a 3D image along 
        a given axis. This image should not have a 4th channel dim.
        Arguments
        ---------
        axis : integer in {0, 1, 2}
            the axis on which to take slices
        reject_zeros : boolean
            whether to reject slices that are all zeros
        """
        self.axis = axis
        self.reject_zeros = reject_zeros

    def __call__(self, x, y=None):
        while True:
            keep_slice  = random.randint(0,x.size(self.axis)-1)
            if self.axis == 0:
                slice_x = x[keep_slice,:,:]
                if y is not None:
                    slice_y = y[keep_slice,:,:]
            elif self.axis == 1:
                slice_x = x[:,keep_slice,:]
                if y is not None:
                    slice_y = y[:,keep_slice,:]
            elif self.axis == 2:
                slice_x = x[:,:,keep_slice]
                if y is not None:
                    slice_y = y[:,:,keep_slice]

            if not self.reject_zeros:
                break
            else:
                if y is not None and th.sum(slice_y) > 0:
                        break
                elif th.sum(slice_x) > 0:
                        break
        if y is not None:
            return slice_x, slice_y
        else:
            return slice_x


class RandomCrop(object):

    def __init__(self, size):
        """
        Randomly crop a torch tensor
        Arguments
        --------
        size : tuple or list
            dimensions of the crop
        """
        self.size = size

    def __call__(self, *inputs):
        h_idx = random.randint(0,inputs[0].size(1)-self.size[0])
        w_idx = random.randint(0,inputs[1].size(2)-self.size[1])
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input[:, h_idx:(h_idx+self.size[0]),w_idx:(w_idx+self.size[1])]
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class SpecialCrop(object):

    def __init__(self, size, crop_type=0):
        """
        Perform a special crop - one of the four corners or center crop
        Arguments
        ---------
        size : tuple or list
            dimensions of the crop
        crop_type : integer in {0,1,2,3,4}
            0 = center crop
            1 = top left crop
            2 = top right crop
            3 = bottom right crop
            4 = bottom left crop
        """
        if crop_type not in {0, 1, 2, 3, 4}:
            raise ValueError('crop_type must be in {0, 1, 2, 3, 4}')
        self.size = size
        self.crop_type = crop_type
    
    def __call__(self, x, y=None):
        if self.crop_type == 0:
            # center crop
            x_diff  = (x.size(1)-self.size[0])/2.
            y_diff  = (x.size(2)-self.size[1])/2.
            ct_x    = [int(math.ceil(x_diff)),x.size(1)-int(math.floor(x_diff))]
            ct_y    = [int(math.ceil(y_diff)),x.size(2)-int(math.floor(y_diff))]
            indices = [ct_x,ct_y]        
        elif self.crop_type == 1:
            # top left crop
            tl_x = [0, self.size[0]]
            tl_y = [0, self.size[1]]
            indices = [tl_x,tl_y]
        elif self.crop_type == 2:
            # top right crop
            tr_x = [0, self.size[0]]
            tr_y = [x.size(2)-self.size[1], x.size(2)]
            indices = [tr_x,tr_y]
        elif self.crop_type == 3:
            # bottom right crop
            br_x = [x.size(1)-self.size[0],x.size(1)]
            br_y = [x.size(2)-self.size[1],x.size(2)]
            indices = [br_x,br_y]
        elif self.crop_type == 4:
            # bottom left crop
            bl_x = [x.size(1)-self.size[0], x.size(1)]
            bl_y = [0, self.size[1]]
            indices = [bl_x,bl_y]
        
        x = x[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]

        if y is not None:
            y = y[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]
            return x, y
        else:
            return x


class Pad(object):

    def __init__(self, size):
        """
        Pads an image to the given size
        Arguments
        ---------
        size : tuple or list
            size of crop
        """
        self.size = size

    def __call__(self, x, y=None):
        x = x.numpy()
        shape_diffs = [int(np.ceil((i_s - d_s))) for d_s,i_s in zip(x.shape,self.size)]
        shape_diffs = np.maximum(shape_diffs,0)
        pad_sizes = [(int(np.ceil(s/2.)),int(np.floor(s/2.))) for s in shape_diffs]
        x = np.pad(x, pad_sizes, mode='constant')
        if y is not None:
            y = y.numpy()
            y = np.pad(y, pad_sizes, mode='constant')
            return th.from_numpy(x), th.from_numpy(y)
        else:
            return th.from_numpy(x)


class RandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.
        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p
        v : boolean
            whether to vertically flip w/ probability p
        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.p = p

    def __call__(self, x, y=None):
        x = x.numpy()
        if y is not None:
            y = y.numpy()
        # horizontal flip with p = self.p
        if self.horizontal:
            if random.random() < self.p:
                x = x.swapaxes(2, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 2)
                if y is not None:
                    y = y.swapaxes(2, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 2)
        # vertical flip with p = self.p
        if self.vertical:
            if random.random() < self.p:
                x = x.swapaxes(1, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 1)
                if y is not None:
                    y = y.swapaxes(1, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 1)
        if y is None:
            # must copy because torch doesnt current support neg strides
            return th.from_numpy(x.copy())
        else:
            return th.from_numpy(x.copy()),th.from_numpy(y.copy())


class RandomOrder(object):
    """
    Randomly permute the channels of an image
    """
    def __call__(self, *inputs):
        order = th.randperm(inputs[0].dim())
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.index_select(0, order)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

