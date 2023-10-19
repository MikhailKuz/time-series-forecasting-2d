from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
import os
import imageio
import matplotlib.pyplot as plt
import torch
import joblib
import torchvision
import random
from copy import deepcopy


def check_filter(targets, candidate, verbose):
    for target in targets:
        flag = True
        
        keys1 = target.keys()
        for c_k in keys1:
            c_v = candidate[c_k]
            if isinstance(c_v, dict) == False:
                if target.get(c_k, None) != c_v:
                    flag = False
                    break
            elif isinstance(c_v, dict):
                keys2 = target[c_k].keys()
                for kk in keys2:
                    vv = c_v[kk]
                    if target.get(c_k, {}).get(kk, "") != vv:
                        flag = False
                        break
        if flag:
            if verbose:
                print(candidate)
            return flag
    return False

def name_parse1(log):
    model_name = [log["model_name"]]
    model_name = model_name + [log["opt_params"]["lr"]]
    model_name = ", ".join([str(x) for x in model_name])
    return model_name

def wrap_image_with_text(images, row_name, col_name, k, plot_horizon=True, add_to_title=""):
    C, H, W = images[0].shape
    img_gen = torchvision.utils.make_grid(images, normalize=False, scale_each=False, range=(0, 1), nrow=len(col_name))
    
    cmap = plt.cm.jet
    image = cmap(img_gen[1, :, :].numpy())
    
    # save the image
    plt.imsave('./temp.png', image)
    img_gen = Image.open("./temp.png")
    os.remove("./temp.png")
    grid = img_gen

    font=ImageFont.truetype("/home/aalanov/mkkuznetsov/image_editing/temp/times.ttf", 20)
    draw = ImageDraw.Draw(grid)
    names = col_name
    W_, H_ = grid.size
    th = 20
    ws, hw = zip(*[draw.textbbox((0, 0), name, font=font)[-2:] for name in names])
    new_img = Image.new('RGB', (W_, H_ + max(hw) + th), (255, 255, 255))

    draw = ImageDraw.Draw(new_img)
    W_, H_ = new_img.size
    new_img.paste(grid, box=(0, max(hw) + th))

    for i, name in enumerate(names):
        draw.text((W * i + (W-ws[i])/2 + i * 2, max(hw) - hw[i] + th // 2), name, font=font, fill='black')

    #########

    new_img = new_img.rotate(-90, expand=True)
    grid = new_img
    names = row_name[::-1]
    W_, H_ = grid.size
    th = 20
    ws, hw = zip(*[draw.textbbox((0, 0), name, font=font)[-2:] for name in names])
    new_img = Image.new('RGB', (W_, H_ + max(hw) + th), (255, 255, 255))

    draw = ImageDraw.Draw(new_img)
    W_, H_ = new_img.size
    new_img.paste(grid, box=(0, max(hw) + th))

    for i, name in enumerate(names):
        draw.text((H * i + (H-ws[i])/2 + i * 2, max(hw) - hw[i] + th // 2), name, font=font, fill='black')

    new_img = new_img.rotate(90, expand=True)

    if plot_horizon:
        grid = new_img
        draw = ImageDraw.Draw(grid)
        names = [add_to_title] + [f"horizon: {k+1}"]
        names = [', '.join(names)]
        W_, H_ = grid.size
        th = 20
        ws, hw = zip(*[draw.textbbox((0, 0), name, font=font)[-2:] for name in names])
        new_img = Image.new('RGB', (W_, H_ + max(hw) + th), (255, 255, 255))

        draw = ImageDraw.Draw(new_img)
        W_, H_ = new_img.size
        new_img.paste(grid, box=(0, 0))

        draw.text((W_ // 2 - ws[0] / 2, H_ - hw[0] - th // 2), names[0], font=font, fill='black')
    return new_img

def gen_gif(out_path="./temp.gif", paths=[], loader="test", target_filter={"model_name": ["EarthFormer"]},
            name_parse=lambda x: x["model_name"], sort_name=lambda x: list(x),
            H=150, W=200, horizon=1, plot_horizon=True, duration=0.2,
            ret_only_ids=False, add_to_title="", verbose=1, out_ration=1):
    dt_to_model_res = {}
    good_exp_ids = []
    
    for path in paths:
        if not os.path.exists(path):
            continue
        
        configs = os.listdir(path)
        for conf in tqdm(configs):
            
            if not os.path.exists(path + conf + "/config.pkl"):
                continue
            
            log = joblib.load(path + conf + "/config.pkl")
            log = {k: v for k, v in log.items() if "_data" not in k and "_path" not in k}
            
            if not check_filter(target_filter, log, verbose):
                continue
            
            good_exp_ids.append(log["exp_id"])
            if ret_only_ids:
                continue
            
            preds = joblib.load(path + conf + '/preds.pkl')
            dt_name = log["data"][5:]
            
            dt_to_model_res[dt_name] = {} if dt_name not in dt_to_model_res else dt_to_model_res[dt_name]
            model_name = name_parse(log)
            dt_to_model_res[dt_name][model_name] = preds[loader]["preds"]
            dt_to_model_res[dt_name]["Target"] = preds[loader]["target"]
            
    if ret_only_ids:
        return good_exp_ids
    
    assert len(dt_to_model_res) > 0
    all_models = list(dt_to_model_res[list(dt_to_model_res.keys())[0]].keys())
    all_data = list(dt_to_model_res.keys())
    
    m_order = ["Linear", "NLinear", "DLinear", "ConvLSTM", "EarthFormer"]
    
    rename_map = {k: ''.join([str(random.randint(0, 9)) for i in range(20)]) for k in m_order}
    rename_map_inverse = {v: k for k, v in rename_map.items()}
    renamed_all_models = []
    
    for x in all_models:
        y = x
        for k, v in rename_map.items():
            y = y.replace(k, v)
        renamed_all_models.append(y)
    
    model_order = []
    
    for model in m_order:
        model_order = model_order + sort_name([x for x in renamed_all_models if rename_map[model] in x])
    
    renamed_all_models = list(renamed_all_models)
    renamed_all_models = [x for x in renamed_all_models if x != "Target"]
    model_order = model_order + sort_name([x for x in renamed_all_models if sum([rename_map[y] in x for y in m_order]) == 0])
    model_order = model_order + ["Target"]
    _model_order = []
    
    for x in model_order:
        y = x
        for k, v in rename_map_inverse.items():
            y = y.replace(k, v)
        _model_order.append(y)
    
    model_order = _model_order
    if verbose:
        print(model_order)
    assert set(model_order) == set(all_models)
    
    data_order = ['Missouri', 'MadhyaPradesh', 'CentralKZ']
    dt_order = []
    
    for dt in data_order:
        dt_order = dt_order + [x for x in all_data if dt in x]
    
    dt_order = dt_order + [x for x in all_data if sum([y in x for y in data_order]) == 0]
    assert set(dt_order) == set(all_data)
    
    
    dt_to_N_out = {dt: torch.load(f"../data/splits/reg/test/pdsi_{dt}.pt").shape[0] for dt in dt_order}
    
    k = horizon - 1
    tr = torchvision.transforms.Resize((H, W))

    images = []
    for dt in dt_order:
        for model in model_order:
            if dt not in dt_to_model_res or model not in dt_to_model_res[dt]:
                img = torch.zeros((dt_to_N_out[dt] - log["seq_len"] - log["pred_len"] + 1, log["pred_len"], H, W))[:, k, ...]
            else:
                min_v = dt_to_model_res[dt]["Target"].min()
                max_v = dt_to_model_res[dt]["Target"].max()
                img = dt_to_model_res[dt][model][:, k, ...]
                img = tr(img)
                img = (img - min_v) / max_v
            
            img = torch.stack([torch.zeros_like(img), img, torch.zeros_like(img)]).transpose(0, 1)
            images.append(img)

    images = [list(x) for x in zip(*images)]
    imageio_imgs = []

    for image in tqdm(images):
        img = wrap_image_with_text(image, dt_order, model_order, k, plot_horizon=plot_horizon, add_to_title=add_to_title)
        if out_ration != 1:
            shape = [int(x * out_ration) for x in img.size]
            img = img.resize(shape, resample=Image.LANCZOS)
        imageio_imgs.append(img)

    imageio.mimsave(out_path, imageio_imgs, duration=duration)
