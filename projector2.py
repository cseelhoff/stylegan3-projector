"""Project given image to the latent space of pretrained network pickle."""
import copy
import os
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import dnnlib
import legacy
import math
import PIL.ImageFilter
import dlib
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def resize_image(image):
    newx = 224
    newy = 224
    if image.shape[2] > newx or image.shape[3] > newy:
        image = F.interpolate(image, size=(newx, newy), mode='area')    
    return image

def calcTargets(coords, target_images, vgg16):
    targets = []
    i = 0
    for c in coords:
        target = resize_image(target_images[0:1, 0:3, c[0]:c[1], c[2]:c[3]])
        PIL.Image.fromarray(target.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy(), 'RGB').save(f'proj{i}.jpg')
        targets.append(vgg16(target, resize_images=False, return_lpips=True))
        i += 1
    return targets

def project2(
    target_pil, eyeleftp, eyerightp, mouthp, rotate_mask,
    device: torch.device, G, vgg16, starting_wplus_space, target_short_name: str,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.03,
    initial_noise_factor       = 0.25,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.85,
    regularize_noise_weight    = 1e5
):
    outdir = "./outdir"
    seed = 303
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_mask = torch.tensor(rotate_mask.transpose([2, 0, 1]), device=device)
    target_pil.save(f'{outdir}/{target_short_name}target.jpg')
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)
    target=torch.mul(target, torch_mask)
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) 
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    minx = 0
    maxx = 1024
    miny = 0
    maxy = 1024
    canvas = [0, 1024, 0, 1024] #y_start,y_end,x_start,x_end
    canvas1 = [0, 512, 0, 512] #y_start,y_end,x_start,x_end
    canvas2 = [512, 1024, 0, 512] #y_start,y_end,x_start,x_end
    canvas3 = [0, 512, 512, 1024] #y_start,y_end,x_start,x_end
    canvas4 = [512, 1024, 512, 1024] #y_start,y_end,x_start,x_end
    mouth = [max(mouthp[1]-140,miny), min(mouthp[1]+80,maxy), max(mouthp[0]-110,minx), min(mouthp[0]+110,maxx)]
    left_eye = [max(eyeleftp[1]-80,miny), min(eyeleftp[1]+80,maxy), max(eyeleftp[0]-140,minx), min(eyeleftp[0]+80,maxx)]
    right_eye = [max(eyerightp[1]-80,miny), min(eyerightp[1]+80,maxy), max(eyerightp[0]-80,minx), min(eyerightp[0]+140,maxx)]
    coords = [mouth, left_eye, right_eye]
    coords = [canvas1, canvas2, canvas3, canvas4, mouth, left_eye, right_eye]
    coords = [canvas, mouth, left_eye, right_eye]
    #coords = [canvas]
    targets = calcTargets(coords, target_images, vgg16)
    #starting_wplus_space = torch.load(f'../restyle/output/inference_coupled/{target_short_name}.pt')
    zs = torch.randn([10000, G.mapping.z_dim], device=device)
    w_stds = G.mapping(zs, None).std(0)
    w_opt = (G.mapping(torch.randn([1,G.mapping.z_dim], device=device), None, truncation_psi=0.0001) - G.mapping.w_avg) / w_stds
    images = G.synthesis(w_opt * w_stds + G.mapping.w_avg)
    tf = Compose([
        Resize(224),
        lambda x: torch.clamp((x+1)/2,min=0,max=1),
    ])
    TF.to_pil_image(tf(images)[0]).save('synth.jpg')
    #starting_wplus_space = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None, truncation_psi=0.0001) - G.mapping.w_avg)
    #w_opt = starting_wplus_space.detach().clone().cuda()
    w_opt.requires_grad = True
    #w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    #optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True
    for step in range(num_steps):
        t = step / num_steps
        #w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        w_noise_scale = initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = ((w_opt + w_noise) * w_stds) + G.mapping.w_avg 
        synth_images = G.synthesis(ws, noise_mode='const')
        if step % 20 == 0:
            TF.to_pil_image(tf(synth_images)[0]).save(f'synth{step}.jpg')
    
        dist = 0
        for (c, target) in zip(coords, targets):
            synth_image_clone = synth_images.clone()
            synth_image_clone = (synth_image_clone + 1) * (255/2)
            synth_image_clone = torch.mul(synth_image_clone, torch_mask)
            synth = resize_image(synth_image_clone[0:1, 0:3, c[0]:c[1], c[2]:c[3]])
            #PIL.Image.fromarray(synth.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy(), 'RGB').save(f'proj.jpg')
            synth_features = vgg16(synth, resize_images=False, return_lpips=True)
            dist += (target - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss2 = w_opt.square().sum() * 0.00003
        loss = loss2 + dist + (reg_loss * regularize_noise_weight)
        #loss = dist
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        #w_out[step] = ws.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    #projected_w_steps = w_out
    # Save final projected frame and W vector.
    #target_pil.save(f'{outdir}/{target_short_name}target.png')
    #projected_w = projected_w_steps[-1]
    projected_w = ws.detach()[0]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    #TF.to_pil_image(tf(synth_image)[0]).save(f'{outdir}/{target_short_name}proj.jpg')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{target_short_name}.proj.png')
    np.savez(f'{outdir}/{target_short_name}.npz', w=projected_w.unsqueeze(0).cpu().numpy())

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def quad_to_rect(quad, x, y):
    Qx0 = quad[0]
    Qy0 = quad[1]
    Qx1 = quad[2]
    Qy1 = quad[3]
    Qx2 = quad[4]
    Qy2 = quad[5]
    Qx3 = quad[6]
    Qy3 = quad[7]
    ax = (x - Qx0) + (Qx1 - Qx0) * (y - Qy0) / (Qy0 - Qy1)
    a3x = (Qx3 - Qx0) + (Qx1 - Qx0) * (Qy3 - Qy0) / (Qy0 - Qy1)
    a2x = (Qx2 - Qx0) + (Qx1 - Qx0) * (Qy2 - Qy0) / (Qy0 - Qy1)
    ay = (y - Qy0) + (Qy3 - Qy0) * (x - Qx0) / (Qx0 - Qx3)
    a1y = (Qy1 - Qy0) + (Qy3 - Qy0) * (Qx1 - Qx0) / (Qx0 - Qx3)
    a2y = (Qy2 - Qy0) + (Qy3 - Qy0) * (Qx2 - Qx0) / (Qx0 - Qx3)
    bx = x * y - Qx0 * Qy0 + (Qx1 * Qy1 - Qx0 * Qy0) * (y - Qy0) / (Qy0 - Qy1)
    b3x = Qx3 * Qy3 - Qx0 * Qy0 + (Qx1 * Qy1 - Qx0 * Qy0) * (Qy3 - Qy0) / (Qy0 - Qy1)
    b2x = Qx2 * Qy2 - Qx0 * Qy0 + (Qx1 * Qy1 - Qx0 * Qy0) * (Qy2 - Qy0) / (Qy0 - Qy1)
    by = x * y - Qx0 * Qy0 + (Qx3 * Qy3 - Qx0 * Qy0) * (x - Qx0) / (Qx0 - Qx3)
    b1y = Qx1 * Qy1 - Qx0 * Qy0 + (Qx3 * Qy3 - Qx0 * Qy0) * (Qx1 - Qx0) / (Qx0 - Qx3)
    b2y = Qx2 * Qy2 - Qx0 * Qy0 + (Qx3 * Qy3 - Qx0 * Qy0) * (Qx2 - Qx0) / (Qx0 - Qx3)

    l = (ax / a3x) + (1 - a2x / a3x) * (bx - b3x * ax / a3x) / (b2x - b3x * a2x / a3x)
    m = (ay / a1y) + (1 - a2y / a1y) * (by - b1y * ay / a1y) / (b2y - b1y * a2y / a1y)
    return l, m

def align_face(filepath, predictor, detector):    
    img2 = dlib.load_rgb_image(filepath)
    dets = detector(img2, 1)
    shape = None
    for k, d in enumerate(dets):
        shape = predictor(img2, d)
    if not shape:
        raise Exception("Could not find face in image! Please try another image!")
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    landmarks = np.array(a)
    img = PIL.Image.open(filepath)
    lm_eye_left = landmarks[36:42]  # left-clockwise
    lm_eye_right = landmarks[42:48]  # left-clockwise
    lm_mouth_outer = landmarks[48:60]  # left-clockwise
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = np.asarray(lm_mouth_outer[0])
    mouth_right = np.asarray(lm_mouth_outer[6])
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    center = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([center - x - y, center - x + y, center + x + y, center + x - y])
    length_of_one_quad_side = np.hypot(*x) * 2
    output_size = 1024    
    # Shrink.
    shrink_multiplier = int(np.floor(length_of_one_quad_side / output_size * 0.5))
    if shrink_multiplier > 1:
        shrink_to_size = (int(np.rint(float(img.size[0]) / shrink_multiplier)), int(np.rint(float(img.size[1]) / shrink_multiplier)))
        img = img.resize(shrink_to_size, PIL.Image.LANCZOS)
        quad /= shrink_multiplier
        length_of_one_quad_side /= shrink_multiplier
        eye_left /= shrink_multiplier
        eye_right /= shrink_multiplier
        mouth_avg /= shrink_multiplier

    # Crop.
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )    
    border = max(int(np.rint(length_of_one_quad_side * 0.1)), 3)
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    if crop_width < img.size[0] or crop_height < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
        eye_left -= crop[0:2]
        eye_right -= crop[0:2]
        mouth_avg -= crop[0:2]

    # Transform.    
    flat_quad = (quad + 0.5).flatten()
    transform_size = 4096
    white_img = np.full([img.size[1], img.size[0], 3], 255, dtype = np.uint8)
    white_img_pil = PIL.Image.fromarray(white_img)
    new_white = white_img_pil.transform((transform_size, transform_size), PIL.Image.Transform.QUAD, flat_quad, PIL.Image.Resampling.BILINEAR)
    small_white = new_white.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)
    rotate_mask = np.array(small_white, dtype=np.uint8) / 255
    img = img.transform((transform_size, transform_size), PIL.Image.Transform.QUAD, flat_quad, PIL.Image.Resampling.BILINEAR)
    img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)

    l, m = quad_to_rect(flat_quad, eye_left[0], eye_left[1])
    l = math.floor(l * output_size)
    m = math.floor(m * output_size)
    eyeleftp = (l, m)
    
    l, m = quad_to_rect(flat_quad, eye_right[0], eye_right[1])
    l = math.floor(l * output_size)
    m = math.floor(m * output_size)
    eyerightp = (l, m)
    
    l, m = quad_to_rect(flat_quad, mouth_avg[0], mouth_avg[1])
    l = math.floor(l * output_size)
    m = math.floor(m * output_size)
    mouthp = (l, m)
    # Save aligned image.
    return img, eyeleftp, eyerightp, mouthp, rotate_mask

import pickle
if __name__ == "__main__":
    device = torch.device('cuda')
    network_pkl = "./stylegan3-t-ffhqu-1024x1024.pkl"
    with dnnlib.util.open_url(network_pkl) as fp:
        #G = pickle.load(fp)['G_ema'].to(device)
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt') as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #predictor = get_model("resnet50_2020-07-20", max_size=2048)
    #predictor.eval()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    starting_wplus_space = torch.load(f'./mean.pt')
    detector = dlib.get_frontal_face_detector()

    target_short_name = '10187.jpg'
    file_path = './raw/' + target_short_name
    for target_short_name in os.listdir('./raw/'):
        print(target_short_name)
        if os.path.isfile(os.path.join('./outdir/', (target_short_name + '.npz'))):
            continue
        file_path = os.path.join('./raw/', target_short_name)
        #try:
        target_pil, eyeleftp, eyerightp, mouthp, rotate_mask = align_face(file_path, predictor, detector)
        target_pil = target_pil.convert("RGB")
#find w+ space, by comparing q0-q3
        project2(target_pil, eyeleftp, eyerightp, mouthp, rotate_mask, device, G, vgg16, starting_wplus_space, target_short_name)
    #except Exception:
    #    continue
