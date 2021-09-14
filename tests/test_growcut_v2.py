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
    """
    # Teeth
    imgpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001.nii.gz"
    labelpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001_label_tri.nii.gz"
    savepath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_python/results"

    img = nib.load(imgpath)
    imgdata = np.squeeze(img.get_fdata()[120:580, 10:350, 225:226])
    label = nib.load(labelpath)
    labeldata = np.squeeze(label.get_fdata()[120:580, 10:350, 225:226])
    seedsdata = np.zeros(labeldata.shape)

    """

    # Pelvis
    imgpath = "/data/tianmu/data/dynamic_segmentation/brats2015/train/all/sample_54686_image_norm_crop_resize.mha"
    labelpath = "/data/tianmu/data/dynamic_segmentation/brats2015/train/all/sample_54686_label_binary_crop_resize.mha"
    savepath = "/data/tianmu/data/dynamic_segmentation/brats2015/train/all_fastgc"

    img = _itk_read_image_from_file(imgpath)
    middle_index = img.GetSize()[2] // 2
    imgdata = _itk_read_array_from_file(imgpath)[middle_index : (middle_index+1), :, :]
    label = _itk_read_image_from_file(labelpath)
    labeldata = _itk_read_array_from_file(labelpath)[middle_index : (middle_index+1), :, :]
    seedsdata = np.zeros(labeldata.shape)

    fg_indices = np.where(labeldata == 1)
    bg_indices = np.where(labeldata == 0)
    num_clicks_fg = 5
    num_clicks_bg = 5
    fg_selected = np.random.choice(fg_indices[0].shape[0], num_clicks_fg, replace=False).tolist()
    bg_selected = np.random.choice(bg_indices[0].shape[0], num_clicks_bg, replace=False).tolist()

    for fg in fg_selected:
        seedsdata[fg_indices[0][fg], fg_indices[1][fg], fg_indices[2][fg]] = 2
    
    for bg in bg_selected:
        seedsdata[bg_indices[0][bg], bg_indices[1][bg], bg_indices[2][bg]] = 1

    # nlabels = np.unique(labeldata)
    # for i in nlabels:
    #     mask = labeldata == i
    #     mask = binary_erosion(mask, structure=np.ones((1,3,3)))
    #     mask = binary_erosion(mask, structure=np.ones((1,3,3)))
    #     mask = binary_erosion(mask, structure=np.ones((1,3,3)))
    #     mask = binary_erosion(mask, structure=np.ones((1,3,3)))
    #     mask = binary_erosion(mask, structure=np.ones((1,3,3)))

    #     seedsdata = seedsdata + mask * (i+1)

    start = time.time()
    distPre, labPre = fastgc(imgdata, seedsdata, newSeg = True, verbose = True)
    end = time.time()
    labPre[labPre == 1] = 0
    labPre[labPre == 2] = 1
    dice_score = _get_dice(labPre, labeldata, epsilon=1e-6, weight=None)

    print("time used:", end - start, "seconds")
    print(f"dice_score = {dice_score}")

    """
    # Teeth
    nib.save(nib.Nifti1Image(imgdata, img.affine), join(savepath, "original_img_teeth.nii.gz"))
    nib.save(nib.Nifti1Image(labeldata, img.affine), join(savepath, "original_label_teeth.nii.gz"))
    nib.save(nib.Nifti1Image(seedsdata, img.affine), join(savepath, "seedsdata_teeth.nii.gz"))
    nib.save(nib.Nifti1Image(labPre, img.affine), join(savepath, "fast_growcut_results_teeth.nii.gz"))
    
    """
    # Pelvis
    # nib.save(nib.Nifti1Image(imgdata, img.affine), join(savepath, "original_img_pelvis.nii.gz"))
    # nib.save(nib.Nifti1Image(labeldata, img.affine), join(savepath, "original_label_pelvis.nii.gz"))
    # nib.save(nib.Nifti1Image(seedsdata, img.affine), join(savepath, "seedsdata_pelvis.nii.gz"))
    # nib.save(nib.Nifti1Image(labPre, img.affine), join(savepath, "fast_growcut_results_pelvis.nii.gz"))


    _itk_write_array_to_file(imgdata, img, os.path.join(savepath, "sample_54686_image_original.mha"))
    _itk_write_array_to_file(labeldata, img, os.path.join(savepath, "sample_54686_label_original.mha"))
    _itk_write_array_to_file(seedsdata, img, os.path.join(savepath, "sample_54686_seed.mha"))
    _itk_write_array_to_file(labPre, img, os.path.join(savepath, "sample_54686_fastgc.mha"))


if __name__ == "__main__":
    test_fastgrowcut()