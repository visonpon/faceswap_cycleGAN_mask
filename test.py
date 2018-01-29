import torch
from networks import *
from PIL import Image
import option as opt
import functools
import face_recognition
from moviepy.editor import VideoFileClip

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
              
    return norm_layer
###################
input_nc=3
output_nc=3
ngf=64
norm = get_norm_layer(norm_type=norm)
no_dropout = False
gpu_ids =0
###################
netG_A = networks.Generator(opt.input_nc, opt.output_nc,opt.ngf, opt.norm, opt.no_dropout,opt.gpu_ids)
netG_B = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.no_dropout,opt.gpu_ids)
netD_A = networks.Discriminator().cuda()
netD_B = networks.Discriminator().cuda()


save_filename = '%s_net_%s.pth' % (epoch_label, network_label)    
try:
    netG_A.load_state_dict(torch.load(os.path.join(self.save_dir, save_filename1)))
    netG_B.load_state_dict(torch.load(os.path.join(self.save_dir, save_filename2)))
    netD_A.load_state_dict(torch.load(os.path.join(self.save_dir, save_filename3)))
    netD_A.load_state_dict(torch.load(os.path.join(self.save_dir, save_filename4)))
    print ("model loaded.")
except:
    print ("Weights file not found.")
    pass 
  

def cycle_variables(input,netG):
    fake_output,mask_output = netG(input)
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)#(batch_size, 64, 64, 1) tensor, mask output of generator_A (netGA).
    rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
    masked_fake_output = alpha * rgb + (1-alpha) * input 
       
    fn_generate = masked_fake_output#A function that takes distorted_A as input and outputs fake_A.
    fn_mask = torch.cat((alpha,alpha,alpha),0)#A function that takes distorted_A as input and outputs mask_A.
    fn_abgr = torch.cat((alpha,rgb),0)#A function that takes distorted_A as input and outputs concat([mask_A, fake_A]).
    return input, fake_output, alpha, fn_generate, fn_mask, fn_abgr

A, fake_A, mask_A, path_A, path_mask_A, path_abgr_A = cycle_variables(input,netG_A)
B, fake_B, mask_B, path_B, path_mask_B, path_abgr_B = cycle_variables(input,netG_B)
real_A = Image.Resize(real_A,(64,64))
real_B = Image.Resize(real_B,(64,64))

################################################################
whom2whom = "BtoA" # default trainsforming faceB to faceA

if whom2whom is "AtoB":
    path_func = path_abgr_B
elif whom2whom is "BtoA":
    path_func = path_abgr_A
else:
    print ("whom2whom should be either AtoB or BtoA")
###############################################################

use_smoothed_mask = True
use_smoothed_bbox = True

def get_smoothed_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    x0 = int(0.65*prev_x0 + 0.35*x0)
    x1 = int(0.65*prev_x1 + 0.35*x1)
    y1 = int(0.65*prev_y1 + 0.35*y1)
    y0 = int(0.65*prev_y0 + 0.35*y0)
    return x0, x1, y0, y1    
    
def set_global_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    prev_x0 = x0
    prev_x1 = x1
    prev_y1 = y1
    prev_y0 = y0

def process_video(input_img):   
    # modify this line to reduce input size
    #input_img = input_img[:, input_img.shape[1]//3:2*input_img.shape[1]//3,:] 
    image = input_img
    faces = face_recognition.face_locations(image, model="cnn")
    
    if len(faces) == 0:
        comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
        comb_img[:, :input_img.shape[1], :] = input_img
        comb_img[:, input_img.shape[1]:, :] = input_img
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        triple_img[:, :input_img.shape[1], :] = input_img
        triple_img[:, input_img.shape[1]:input_img.shape[1]*2, :] = input_img      
        triple_img[:, input_img.shape[1]*2:, :] = (input_img * .15).astype('uint8')
    
    mask_map = np.zeros_like(image)
    
    global prev_x0, prev_x1, prev_y0, prev_y1
    global frames    
    for (x0, y1, x1, y0) in faces:        
        # smoothing bounding box
        if use_smoothed_bbox:
            if frames != 0:
                x0, x1, y0, y1 = get_smoothed_coord(x0, x1, y0, y1)
                set_global_coord(x0, x1, y0, y1)
            else:
                set_global_coord(x0, x1, y0, y1)
                frames += 1
        
        h = x1 - x0
        w = y1 - y0
            
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        roi_image = cv2_img[x0+h//15:x1-h//15,y0+w//15:y1-w//15,:]
        roi_size = roi_image.shape  
        
        # smoothing mask
        if use_smoothed_mask:
            mask = np.zeros_like(roi_image)
            mask[h//15:-h//15,w//15:-w//15,:] = 255
            mask = cv2.GaussianBlur(mask,(15,15),10)
            orig_img = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        
        ae_input = cv2.resize(roi_image, (64,64))/255. * 2 - 1        
        result = np.squeeze(np.array([path_func([[ae_input]])]))
        result_a = result[:,:,0] * 255
        result_bgr = np.clip( (result[:,:,1:] + 1) * 255 / 2, 0, 255 )
        result_a = cv2.GaussianBlur(result_a ,(7,7),6)
        result_a = np.expand_dims(result_a, axis=2)
        result = (result_a/255 * result_bgr + (1 - result_a/255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        mask_map[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = np.expand_dims(cv2.resize(result_a, (roi_size[1],roi_size[0])), axis=2)
        mask_map = np.clip(mask_map + .15 * input_img, 0, 255 )
        
        result = cv2.resize(result, (roi_size[1],roi_size[0]))
        comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
        comb_img[:, :input_img.shape[1], :] = input_img
        comb_img[:, input_img.shape[1]:, :] = input_img
        
        if use_smoothed_mask:
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = mask/255*result + (1-mask/255)*orig_img
        else:
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = result
            
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        triple_img[:, :input_img.shape[1]*2, :] = comb_img
        triple_img[:, input_img.shape[1]*2:, :] = mask_map
    
    # ========== Change rthe following line to ==========
    return comb_img[:, input_img.shape[1]:, :]  # return result image only
    # return comb_img  # return input and result image combined as one
    #return triple_img #return input,result and mask heatmap image combined as one

#########################################################
global prev_x0, prev_x1, prev_y0, prev_y1
global frames
prev_x0 = prev_x1 = prev_y0 = prev_y1 = 0
frames = 0

output = './ouputdir/OUTPUT_VIDEO.mp4'
clip1 = VideoFileClip("./input_dir/INPUT_VIDEO.mp4")
clip = clip1.fl_image(process_video)#.subclip(11, 13) #NOTE: this function expects color images!!
#########################################################
