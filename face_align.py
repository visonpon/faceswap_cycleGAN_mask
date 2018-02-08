import face_recognition
import cv2
import numpy as np

use_smoothed_mask = True
apply_face_aln = True
use_poisson_blending = False # SeamlessClone is NOT recommended for video.
use_comp_video = True # output a comparison video before/after face swap
use_smoothed_bbox = True

def is_higher_than_480p(x):
    return (x.shape[1] >= 858 and x.shape[0] >= 480)

def is_higher_than_720p(x):
    return (x.shape[1] >= 1280 and x.shape[0] >= 720)

def is_higher_than_1080p(x):
    return (x.shape[1] >= 1920 and x.shape[0] >= 1080)

def calibrate_coord(faces, video_scaling_factor):
    for i, (x0, y1, x1, y0) in enumerate(faces):
        faces[i] = (x0*video_scaling_factor, y1*video_scaling_factor, 
                    x1*video_scaling_factor, y0*video_scaling_factor)
    return faces

def get_faces_bbox(image, model="cnn"):  
    if is_higher_than_1080p(image):
        video_scaling_factor = 4 + video_scaling_offset
        resized_image = cv2.resize(image, 
                                   (image.shape[1]//video_scaling_factor, image.shape[0]//video_scaling_factor))
        faces = face_recognition.face_locations(resized_image, model=model)
        faces = calibrate_coord(faces, video_scaling_factor)
    elif is_higher_than_720p(image):
        video_scaling_factor = 3 + video_scaling_offset
        resized_image = cv2.resize(image, 
                                   (image.shape[1]//video_scaling_factor, image.shape[0]//video_scaling_factor))
        faces = face_recognition.face_locations(resized_image, model=model)
        faces = calibrate_coord(faces, video_scaling_factor)  
    elif is_higher_than_480p(image):
        video_scaling_factor = 2 + video_scaling_offset
        resized_image = cv2.resize(image, 
                                   (image.shape[1]//video_scaling_factor, image.shape[0]//video_scaling_factor))
        faces = face_recognition.face_locations(resized_image, model=model)
        faces = calibrate_coord(faces, video_scaling_factor)
    else:
        faces = face_recognition.face_locations(image, model=model)
    return faces

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
    
def extract_eye_center(shape):
    xs = 0
    ys = 0
    for pnt in shape:
        xs += pnt[0]
        ys += pnt[1]
    return ((xs//6), ys//6)

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M, (xc, yc), angle

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return 90
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotated_img(img, det):
    #print (det, img.shape)
    shape = face_recognition.face_landmarks(img, det)
    pnts_left_eye = shape[0]["left_eye"]
    pnts_right_eye = shape[0]["right_eye"]
    if len(pnts_left_eye) == 0 or len(pnts_right_eye) == 0:
        return img, None, None    
    le_center = extract_eye_center(shape[0]["left_eye"])
    re_center = extract_eye_center(shape[0]["right_eye"])
    M, center, angle = get_rotation_matrix(le_center, re_center)
    M_inv = cv2.getRotationMatrix2D(center, -1*angle, 1)    
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)    
    return rotated, M, M_inv, center

def process_video(input_img):   
    image = input_img
    # ========== Decrease image size if getting memory error ==========
    #image = input_img[:3*input_img.shape[0]//4, :, :]
    #image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2))
    orig_image = np.array(image)
    faces = get_faces_bbox(image, model="cnn")
    
    if len(faces) == 0:
        comb_img = np.zeros([orig_image.shape[0], orig_image.shape[1]*2,orig_image.shape[2]])
        comb_img[:, :orig_image.shape[1], :] = orig_image
        comb_img[:, orig_image.shape[1]:, :] = orig_image
        if use_comp_video:
            return comb_img
        else:
            return image
    
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
                
        if apply_face_aln:
            do_back_rot = True
            image, M, M_inv, center = get_rotated_img(image, [(x0, y1, x1, y0)])
            if M is None:
                do_back_rot = False
        
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        roi_image = cv2_img[x0+h//15:x1-h//15, y0+w//15:y1-w//15, :]
        roi_size = roi_image.shape            
        
        if use_smoothed_mask:
            mask = np.zeros_like(roi_image)
            #print (roi_image.shape, mask.shape)
            mask[h//15:-h//15,w//15:-w//15,:] = 255
            mask = cv2.GaussianBlur(mask,(15,15),10)
            roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        
        ae_input = cv2.resize(roi_image, (64,64))/255. * 2 - 1        
        result = np.squeeze(np.array([path_abgr_A([[ae_input]])]))
        result_a = result[:,:,0] * 255
        result_bgr = np.clip( (result[:,:,1:] + 1) * 255 / 2, 0, 255 )
        result_a = cv2.GaussianBlur(result_a ,(7,7),6)
        result_a = np.expand_dims(result_a, axis=2)
        result = (result_a/255 * result_bgr + (1 - result_a/255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = cv2.resize(result, (roi_size[1],roi_size[0]))        
        result_img = np.array(orig_image)
        
        if use_smoothed_mask and not use_poisson_blending:
            image[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = mask/255*result + (1-mask/255)*roi_image_rgb
        elif use_poisson_blending:
            c = (y0+w//2, x0+h//2)
            image = cv2.seamlessClone(result, image, mask, c, cv2.NORMAL_CLONE)     
            
        if do_back_rot:
            image = cv2.warpAffine(image, M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
            result_img[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = image[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:]
        else:
            result_img[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = image[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:]   

        if use_comp_video:
            comb_img = np.zeros([orig_image.shape[0], orig_image.shape[1]*2,orig_image.shape[2]])
            comb_img[:, :orig_image.shape[1], :] = orig_image
            comb_img[:, orig_image.shape[1]:, :] = result_img
            
    if use_comp_video:
        return comb_img
    else:
        return result_img
