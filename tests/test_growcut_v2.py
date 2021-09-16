""" Tests for the growcut module """

import numpy as np
import SimpleITK as sitk
import time
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from modules.growcut_cpu import fastgc
import os

def _itk_read_image_from_file(image_path):
    return sitk.ReadImage(image_path, sitk.sitkFloat32)

def _itk_read_array_from_file(image_path):
    return sitk.GetArrayFromImage(_itk_read_image_from_file(image_path))

def _itk_write_array_to_file(image_array, ref_image, output_path):
    itk_image = sitk.GetImageFromArray(image_array)
    itk_image.SetSpacing(ref_image.GetSpacing())
    itk_image.SetOrigin(ref_image.GetOrigin())
    itk_image.SetDirection(ref_image.GetDirection())
    sitk.WriteImage(itk_image, output_path, True)

def _get_dice(input, target, epsilon=1e-6, weight=None):

    # input and target shapes must match
    assert weight is None
    assert input.shape == target.shape, "'input' and 'target' must have the same shape"

    # input = flatten(input)
    # target = flatten(target)
    input = input.astype(np.float32)
    target = target.astype(np.float32)

    # compute per channel Dice Coefficient
    intersect = (input * target).sum()
    # if weight is not None:
    #     intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = input.sum() + target.sum()
    return 2 * (intersect / np.clip(denominator, a_min=epsilon, a_max=None))

def test_fastgrowcut():

    imgpath = "/data/tianmu/data/dynamic_segmentation/brats2015/train/all/sample_35537_image_norm_crop_resize.mha"
    labelpath = "/data/tianmu/data/dynamic_segmentation/brats2015/train/all/sample_35537_label_binary_crop_resize.mha"
    savepath = "/data/tianmu/data/dynamic_segmentation/brats2015/train/all_fastgc"

    img = _itk_read_image_from_file(imgpath)
    label = _itk_read_image_from_file(labelpath)
    
    # middle_index = img.GetSize()[2] // 2
    # imgdata = _itk_read_array_from_file(imgpath)[middle_index // 2 : (middle_index+1), :, :]    
    # labeldata = _itk_read_array_from_file(labelpath)[middle_index // 2 : (middle_index+1), :, :]

    imgdata = _itk_read_array_from_file(imgpath)
    labeldata = _itk_read_array_from_file(labelpath)
        
    fg_indices = np.where(labeldata == 1)
    bg_indices = np.where(labeldata == 0)
    existing_idx = set()

    for _ in range(5):
        num_clicks_fg = 2
        num_clicks_bg = 3
        fg_selected = np.random.choice(fg_indices[0].shape[0], num_clicks_fg, replace=False).tolist()
        bg_selected = np.random.choice(bg_indices[0].shape[0], num_clicks_bg, replace=False).tolist()

        for fg in fg_selected:
            curr_choice = (fg_indices[0][fg], fg_indices[1][fg], fg_indices[2][fg])
            if curr_choice not in existing_idx:
                existing_idx.add(curr_choice)
        
        for bg in bg_selected:
            curr_choice = (bg_indices[0][bg], bg_indices[1][bg], bg_indices[2][bg])
            if curr_choice not in existing_idx:
                existing_idx.add(curr_choice)

        seedsdata = np.zeros(labeldata.shape)
        for idx in existing_idx:
            if labeldata[idx] == 1:
                seedsdata[idx] = 2
            else:
                seedsdata[idx] = 1

        start = time.time()
        distPre, labPre = fastgc(imgdata, seedsdata, newSeg = True, verbose = True)
        end = time.time()
        labPre[labPre == 1] = 0
        labPre[labPre == 2] = 1
        dice_score = _get_dice(labPre, labeldata, epsilon=1e-6, weight=None)

        print("time used:", end - start, "seconds")
        print(f"dice_score = {dice_score}")
        n_pos_seed = (seedsdata == 2).sum()
        n_neg_seed = (seedsdata == 1).sum()
        print(f"pos_seed = {n_pos_seed}, neg_seed = {n_neg_seed}, total = {len(existing_idx)}")

    # _itk_write_array_to_file(imgdata, img, os.path.join(savepath, "sample_35537_image_original.mha"))
    # _itk_write_array_to_file(labeldata, img, os.path.join(savepath, "sample_35537_label_original.mha"))
    # _itk_write_array_to_file(seedsdata, img, os.path.join(savepath, "sample_35537_seed.mha"))
    # _itk_write_array_to_file(labPre, img, os.path.join(savepath, "sample_35537_fastgc.mha"))


if __name__ == "__main__":
    test_fastgrowcut()