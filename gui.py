import os
import torch
import torchvision
#from models.stylegan2.model import Generator
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
import torch
import torchvision
import torchvision.transforms as T
import dnnlib
import legacy

models = {}
frames = {}
positive = None

def load_npz_func():
    npz_dir = npz_dir_entry.get()
    a = 0
    file_names = os.listdir(npz_dir)
    file_names.sort()
    for file_name in file_names:
        if file_name.endswith(".npz"):
            file_path = os.path.join(npz_dir, file_name)
            if file_path not in models.keys():
                a += 1
                if a > 799:
                    break
                print(file_path)
                weights = torch.from_numpy(np.load(file_path)['w']).cuda()
                models[file_path] = torch.squeeze(weights)
                with torch.no_grad():
                    img_orig2 = G.synthesis(weights, noise_mode='const')
                    #img_orig2, _ = g_ema([weights], input_is_latent=True, randomize_noise=False)
                img_orig2 = torchvision.utils.make_grid(img_orig2, normalize=True, scale_each=True, value_range=(-1, 1))
                img_orig2 = torch.squeeze(img_orig2)
                image2 = transform(img_orig2)
                image2 = image2.resize((256, 256), Image.Resampling.LANCZOS)
                photo_image2 = ImageTk.PhotoImage(image2)
                frame = Frame(window, background='black', padx=5, pady=5)
                label = Label(frame, image=photo_image2, background="black", padx=1, pady=1)
                label.image = photo_image2
                label.bind('<Button-1>', lambda event, f=frame: mouse_click(f))
                label.bind('<Button-2>', lambda event, f=frame: middle_click(f))
                label.bind('<Button-3>', lambda event, f=frame: right_click(f))
                label.pack(padx=1, pady=1)
                text.configure(state="normal")
                text.window_create("insert", window=frame, padx=1, pady=1)
                text.configure(state="disabled")
                frames[frame] = file_path


def mouse_click(f):
    if f['background'] != '#000fff000':
        f['background'] = '#000fff000'
    else:
        f['background'] = 'black'
    reload_previews()


def middle_click(f):
    global orig_latent
    orig_latent = models[frames[f]]
    reload_previews()


def right_click(f):
    if f['background'] != 'red':
        f['background'] = 'red'
    else:
        f['background'] = 'black'
    reload_previews()


def slider_move(e):
    reload_previews()


def get_latent():
    latent_code = orig_latent.clone()

    positive_weights = []
    for frame, file_path in frames.items():
        if frame['background'] == '#000fff000':
            positive_weights.append(models[file_path])
    if (len(positive_weights)) == 0:
        positive_weights.append(latent_code)
    tensor_pos_weights = torch.Tensor(len(positive_weights), 18, 512).cuda()
    for i in range(len(positive_weights)):
        tensor_pos_weights[i] = positive_weights[i]
    #torch.cat(positive_weights, out=tensor_pos_weights)
    _, pos_mean = torch.std_mean(tensor_pos_weights, unbiased=False, axis=0)
    
    negative_weights = []
    for frame, file_path in frames.items():
        if frame['background'] == 'red':
            negative_weights.append(models[file_path])
    if (len(negative_weights)) == 0:
        negative_weights.append(latent_code)
    tensor_neg_weights = torch.Tensor(len(negative_weights), 18, 512).cuda()
    for i in range(len(negative_weights)):
        tensor_neg_weights[i] = negative_weights[i]
    #torch.cat(negative_weights, out=tensor_neg_weights)
    _, neg_mean = torch.std_mean(tensor_neg_weights, unbiased=False, axis=0)

    diff = pos_mean - neg_mean
    for i in range(divisions_per_channel * 18):            
        j = i // divisions_per_channel
        m = i % divisions_per_channel
        sfrom = m * settings_per_division
        sto = sfrom + settings_per_division
        diff[j, sfrom:sto] *= c_sliders[i].get()
    diff *= mul_slider.get()
    latent_code += diff
    return latent_code, pos_mean, neg_mean

def reload_previews():
    latent_code, pos_mean, neg_mean = get_latent()
    set_image(latent_code, preview_label, size=1024)
    set_image(pos_mean, preview_label2, size=1024)
    set_image(neg_mean, preview_label3, size=1024)

def save_preview_click(i):
    npz_dir = npz_dir_entry.get()
    latent_code, pos_mean, neg_mean = get_latent()
    latents = [latent_code, pos_mean, neg_mean]
    np.savez(npz_dir + '_.npz', w=latents[i].unsqueeze(0).cpu().detach().numpy())

def set_image(latent_code, label, size=256):
    latent_code = torch.unsqueeze(latent_code, 0)
    with torch.no_grad():
        img_orig2 = G.synthesis(latent_code, noise_mode='const')
        #img_orig2, _ = g_ema([latent_code], input_is_latent=True, randomize_noise=False)
    img_orig2 = torchvision.utils.make_grid(img_orig2, normalize=True, scale_each=True, value_range=(-1, 1))
    img_orig2 = torch.squeeze(img_orig2)
    image2 = transform(img_orig2)
    image2 = image2.resize((size, size), Image.Resampling.LANCZOS)
    photo_image2 = ImageTk.PhotoImage(image2)
    label.configure(image=photo_image2)
    label.image = photo_image2

window = Tk()
toolbar = Frame(window)
text = Text(window, wrap="word", background="black", yscrollcommand=lambda *args: vsb.set(*args))
vsb = Scrollbar(window, command=text.yview)
toolbar.pack(side="top", fill="x")
vsb.pack(side="right", fill="y")
text.pack(side="left", fill="both", expand=True)

left_toolbar_frame = Frame(toolbar)
left_toolbar_frame.pack(side=LEFT)

npz_dir_entry = Entry(left_toolbar_frame, width=15)
npz_dir_entry.insert(END, './outdir')
npz_dir_entry.pack(side=TOP)  # , fill=X, expand=1)

load_npz_button = Button(left_toolbar_frame, text="Load .npz", command=load_npz_func)
load_npz_button.pack(side=TOP)

mul_slider = Scale(left_toolbar_frame, from_=-3, to=3, resolution=0.01, orient=HORIZONTAL, length=100)
mul_slider.bind('<B1-Motion>', slider_move)
mul_slider.bind('<ButtonRelease-1>', slider_move)
mul_slider.set(1)
mul_slider.pack(side=TOP)

toolbar_frames = []
for t in range(9):
    left_toolbar_frame = Frame(toolbar)
    left_toolbar_frame.pack(side=LEFT)
    toolbar_frames.append(left_toolbar_frame)

c_sliders = []

divisions_per_channel = 2
settings_per_division = 512 // divisions_per_channel
for t in toolbar_frames:
    for i in range(2 * divisions_per_channel):
        c_slider = Scale(t, from_=-1, to=3, resolution=0.01, orient=HORIZONTAL, length=100)
        c_slider.bind('<B1-Motion>', slider_move)
        c_slider.bind('<ButtonRelease-1>', slider_move)
        c_slider.set(1)
        c_slider.pack(side=TOP)
        c_sliders.append(c_slider)

# # update color only - 0 shape changes
for i in range(0 * divisions_per_channel, 10 * divisions_per_channel, 1):
    c_sliders[i].set(0)
for i in range(10 * divisions_per_channel, 18 * divisions_per_channel, 1):
    c_sliders[i].set(1)


# # medium sytle updates
# for i in range(0 * divisions_per_channel, 5 * divisions_per_channel, 1):
#     c_sliders[i].set(0)
# for i in range(5 * divisions_per_channel, 7 * divisions_per_channel, 1):
#     c_sliders[i].set(.25)
# for i in range(49,51,1): #eyes direction
#    c_sliders[i].set(0) #eyes direction
# for i in range(52,55,1): #eyes direction
#    c_sliders[i].set(0) #eyes direction
# for i in range(7 * divisions_per_channel, 13 * divisions_per_channel, 1):
#     c_sliders[i].set(.5)
# for i in range(13 * divisions_per_channel, 18 * divisions_per_channel,1):
#     c_sliders[i].set(1)

# for i in range(12):
#     c_sliders[i].set(0)
# #for i in range(12,16,1):
#     #c_sliders[i].set(1)
# for i in range(12,20,1):
#     c_sliders[i].set(.25)
# for i in range(20,44,1):
#     c_sliders[i].set(1)
# for i in range(44,52,1):
#     c_sliders[i].set(1)
# for i in range(52,56,1):
#     c_sliders[i].set(1)
# for i in range(56,72,1):
#     c_sliders[i].set(0)

# # just color sytle updates
# for i in range(12,40,1):
#     c_sliders[i].set(0)
# for i in range(40,56,1):
#     c_sliders[i].set(.5)
# for i in range(56,72,1):
#     c_sliders[i].set(1)

# # # medium sytle updates
# for i in range(0,20,1):
#     c_sliders[i].set(0)
# for i in range(20,28,1):
#     c_sliders[i].set(.25)
# for i in range(25,28,1): #eyes direction
#     c_sliders[i].set(0) #eyes direction
# for i in range(28,52,1):
#     c_sliders[i].set(.5)
# for i in range(52,72,1):
#     c_sliders[i].set(1)

# # change color and focus
# for i in range(0,24,1):
#     c_sliders[i].set(0)
# for i in range(24,44,1):
#     c_sliders[i].set(.5)
# for i in range(44,53,1):
#     c_sliders[i].set(1)
# c_sliders[53].set(.75)
# c_sliders[54].set(.5)
# c_sliders[55].set(.25)
# for i in range(56,72,1):
#     c_sliders[i].set(1)


# full
#for i in range(0,72,1):
#    c_sliders[i].set(1)
    
# fix camera distortion
#for i in range(12,14,1):
#    c_sliders[i].set(1)
#c_sliders[18].set(1)
#c_sliders[20].set(.5)

device = torch.device('cuda')
network_pkl = "./ffhq.pkl"
#network_pkl = "./stylegan3-t-ffhqu-1024x1024.pkl"
with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

#g_ema = Generator(1024, 512, 8)
#g_ema.load_state_dict(torch.load("./stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
G.eval()
G = G.cuda()
transform = T.ToPILImage()

#orig_latent = g_ema.mean_latent(4096).repeat(1, 18, 1)
z_samples = torch.randn([10000, G.mapping.z_dim], device=device)
w_stds = G.mapping(z_samples, None).std(0)
orig_latent = G.mapping(torch.randn([1,G.mapping.z_dim], device=device), None, truncation_psi=0.0001)
w_opt2 = orig_latent - G.mapping.w_avg
w_opt3 = w_opt2 / w_stds

with torch.no_grad():
    img_orig = G.synthesis(orig_latent, noise_mode='const')
    #img_orig, _ = g_ema([orig_latent], input_is_latent=True, randomize_noise=False)
orig_latent = torch.squeeze(orig_latent)
img_orig = torchvision.utils.make_grid(img_orig, normalize=True, scale_each=True, range=(-1, 1))
img_orig = torch.squeeze(img_orig)
image = transform(img_orig)
image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
photo_image = ImageTk.PhotoImage(image)
preview_label = Label(toolbar, image=photo_image, width=1024, height=1024, background='black')
preview_label.image = photo_image
preview_label.bind('<Button-1>', lambda event: save_preview_click(0))
preview_label.pack(side=LEFT)
preview_label2 = Label(toolbar, image=photo_image, width=1024, height=1024, background='black')
preview_label2.image = photo_image
preview_label2.bind('<Button-1>', lambda event: save_preview_click(1))
preview_label2.pack(side=LEFT)
preview_label3 = Label(toolbar, image=photo_image, width=1024, height=1024, background='black')
preview_label3.image = photo_image
preview_label3.bind('<Button-1>', lambda event: save_preview_click(2))
preview_label3.pack(side=LEFT)

window.mainloop()
