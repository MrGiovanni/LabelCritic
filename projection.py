import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random 
import scipy.ndimage as ndimage
import ast
import torch
from PIL import Image
import shutil
import torch.nn.functional as F
import time
import multiprocessing


def plot_organ_projection(list_of_array, organ_name, pid, axis=2,
                           pngpath=None, th=0.5, ct=False, save=True,
                           window='organs'):
    if axis == 2:
        projection = np.zeros((list_of_array[0][:,:,0].shape), dtype='float')
    elif axis == 1:
        projection = np.zeros((list_of_array[0][:,0,:].shape), dtype='float')
    elif axis == 0:
        projection = np.zeros((list_of_array[0][0,:,:].shape), dtype='float')
    else:
        raise ValueError('Axis should be 0, 1, or 2')

    for i in range(len(list_of_array)):
        x=list_of_array[i]
        if ct:
            if window=='organs':
                x=np.where(x>250,250,x)
                x=np.where(x<-150,-150,x)
                x=x+150
                x=x/400
            elif window=='bone':
                x=np.where(x>1500,1500,x)
                x=np.where(x<-500,-500,x)
                x=x+500
                x=x/2000
            else:
                raise ValueError('Window should be organs or bone')
        organ_projection = np.sum(x, axis=axis) * 1.0
        organ_projection /= np.max(organ_projection)
        projection += organ_projection
    
    projection /= np.max(projection)

    if th>0:
        projection=np.where(projection>0,projection/(1/(1-th))+th,0)

    projection *= 255.0
    projection = np.rot90(projection)

    if save:
        if not os.path.exists(pngpath):
            os.makedirs(pngpath)
        cv2.imwrite(os.path.join(pngpath, pid + '_axis_'+str(axis)+'.png'), projection)

        print('Organ projection of ' + organ_name + ' for patient ' + pid + ' is saved to ' + os.path.join(pngpath, pid + '.png'))

    return projection

def plot_organ_projection_3_axis(list_of_array, organ_name, pid,
                                  pngpath=None, th=0.5, ct=False, save=True,
                                  window='organs'):
    projections=[]
    for axis in range(3):
        projections.append(plot_organ_projection(list_of_array, organ_name, pid, axis, 
                                                 pngpath, th=th, ct=ct,save=save,
                                                 window=window))
    return projections




def resample_image(image, original_spacing, target_spacing=(1, 1, 1),order=1):
    """
    Resample the image to the target spacing.

    Parameters:
    image (nibabel.Nifti1Image): Input image to resample.
    target_spacing (tuple): Target spacing in x, y, z directions.

    Returns:
    numpy.ndarray: Resampled image data.
    """
    # Get original spacing
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(image.shape * resize_factor).astype(int)

    # Resample image
    try:image=image.get_fdata()
    except:pass
    resampled_image = ndimage.zoom(image, resize_factor, order=order)

    return resampled_image   


def load_ct_and_mask(pid, organ, datapath,
                     ct_path=None, mask_path=None,
                     resize=True):
    """
    Load and reorient the CT scan and its corresponding mask to the standard LAS orientation.

    Parameters:
        pid (str): Patient ID.
        organ (str): Name of the organ for the mask file.
        datapath (str): Path to the dataset.
        ct_path (str): Path to the CT scan.
        mask_path (str): Path to the mask file.

    Returns:
        ct (np.ndarray): Reoriented CT scan data.
        mask (np.ndarray): Reoriented mask data.
    """
    # Load the CT scan
    if ct_path is None:
        ct_path = os.path.join(datapath, pid, 'ct.nii.gz')
    ct_nii = nib.load(ct_path)
    spacing=ct_nii.header.get_zooms()
    resampled = resample_image(ct_nii, spacing, target_spacing=(1, 1, 1),order=1)
    ct_nii = nib.Nifti1Image(resampled, affine=ct_nii.affine, header=ct_nii.header)

    # Calculate the orientation transformation based on the CT scan
    transform = get_orientation_transform(ct_nii)

    # Apply the transformation to the CT scan data
    ct = apply_transform(ct_nii.get_fdata(), transform)

   # Load the mask using the same transformation
    if mask_path is None:
        # Try the 'predictions' path first
        mask_path = os.path.join(datapath, pid, 'predictions', organ + '.nii.gz')
        
        # If the file doesn't exist in 'predictions', check 'segmentations'
        if not os.path.exists(mask_path):
            mask_path = os.path.join(datapath, pid, 'segmentations', organ + '.nii.gz')

    mask_nii = nib.load(mask_path)

    resampled = resample_image(mask_nii, spacing, target_spacing=(1, 1, 1),order=0)
    if not np.array_equal(resampled, resampled.astype(bool)):
        resampled = np.where(resampled > 0.5, 1, 0)
    mask_nii = nib.Nifti1Image(resampled, affine=mask_nii.affine, header=mask_nii.header)

    # Apply the same transformation to the mask data
    #mask = apply_transform(mask_nii.get_fdata(), transform).astype(np.uint8)
    mask = apply_transform(mask_nii.get_fdata(), get_orientation_transform(mask_nii)).astype(np.uint8)

    return ct, mask
def get_orientation_transform(nii, orientation=('L', 'A', 'S')):
    """
    Compute the transformation needed to reorient the image to LAS standard orientation.
    """
    current_orientation = nib.orientations.io_orientation(nii.affine)
    standard_orientation = nib.orientations.axcodes2ornt(orientation)
    transform = nib.orientations.ornt_transform(current_orientation, standard_orientation)
    return transform

def apply_transform(data, transform):
    """
    Apply the orientation transformation to the image data.
    """
    return nib.orientations.apply_orientation(data, transform)

def apply_spacing_transform(spacing, transform):
    """
    Apply the orientation transformation to the spacing values.
    """
    # Rearrange the spacing based on the reordering in the transform matrix
    reoriented_spacing = [spacing[int(t[0])] for t in transform]
    
    # Take into account the flips (axis direction changes) indicated by the transform
    reoriented_spacing = [sp if t[1] == 1 else -sp for sp, t in zip(reoriented_spacing, transform)]
    
    return tuple(np.abs(reoriented_spacing))

def load_ct(pid, datapath, ct_path, device='cuda'):
    """
    Load and reorient the CT scan to the standard LAS orientation.

    Parameters:
        pid (str): Patient ID.
        datapath (str): Path to the dataset.
        ct_path (str): Path to the CT scan.

    Returns:
        ct (np.ndarray): Reoriented CT scan data.
        spacing (tuple): Reoriented spacing values.
    """
    # Load the CT scan
    if ct_path is None:
        ct_path = os.path.join(datapath, pid, 'ct.nii.gz')
    ct_nii = nib.load(ct_path)
    
    # Get original spacing
    original_spacing = ct_nii.header.get_zooms()

    # Calculate the orientation transformation based on the CT scan
    transform = get_orientation_transform(ct_nii, orientation=('L', 'A', 'S'))

    # Apply the transformation to the CT scan data
    ct = apply_transform(ct_nii.get_fdata(), transform)

    # Apply the transformation to the spacing
    reoriented_spacing = apply_spacing_transform(original_spacing, transform)

    # Move to tensor and device
    ct = torch.from_numpy(ct.copy()).float()
    if device != 'cpu':
        ct = ct.to(device)

    return ct, reoriented_spacing

def window_ct(ct):
    """
    ct: torch.Tensor of shape (D, H, W)
    """
    ct = ct.clone()

    windows={'organs':(-150.0,250.0),
             'bone':(-500.0,1500.0),
             'skeleton':(400.0,2000.0)}

    cts={}
    for window in windows:
        lower_limit, upper_limit = windows[window]
        cts[window] = torch.clamp(ct, min=lower_limit, max=upper_limit)
        cts[window] = (cts[window] - lower_limit) / (upper_limit - lower_limit)
        
    return cts


def project_cts(cts, spacing, axis=1):
    """
    Projects CT scans along a given axis, normalizes the projection, and resamples 
    the remaining two dimensions to have 1x1 mm spacing.

    Parameters:
    cts : dict
        Dictionary of CT scans, where each key represents a window (or scan), 
        and the value is a PyTorch tensor representing the CT scan.
    spacing : tuple or list
        A tuple containing the voxel spacing in each dimension (x, y, z).
    axis : int
        The axis along which to sum the image (default is 2).
    
    Returns:
    resampled_cts : dict
        The dictionary of resampled CT scans with 1x1 mm spacing in the remaining two dimensions.
    """

    # Identify the two remaining dimensions after summing
    remaining_axis = [item for item in range(3) if item != axis]

    # List to store normalized images before resampling
    normalized_images = []

    # Iterate over the CT windows and process each one
    for window in cts:
        # Sum along the specified axis
        summed_image = torch.sum(cts[window], dim=axis)

        # Normalize the image by dividing by the maximum value of the summed image
        normalized_image = summed_image.unsqueeze(0).unsqueeze(0) / torch.max(summed_image)  # Normalize, and add batch and channel dimensions

        normalized_images.append(normalized_image)

    # Stack all normalized images for batch processing
    stacked_images = torch.cat(normalized_images, dim=0)

    # Get the shape of the remaining dimensions (same for all images)
    remaining_dims = stacked_images.shape[-2:]  # Get height and width

    # Calculate the new size (in pixels) to achieve 1x1 mm spacing in the remaining dimensions
    remaining_spacing = [spacing[remaining_axis[i]] for i in range(2)]
    new_size = [int(remaining_dims[i] * remaining_spacing[i]) for i in range(2)]

    print('New size:',new_size)

    # Resample using bilinear interpolation to 1x1 mm spacing
    resampled_images = F.interpolate(stacked_images, size=new_size, mode='bilinear', align_corners=False)

    # Squeeze the batch dimension and store the resampled images in the result dictionary
    resampled_cts = {window: resampled_images[i].squeeze(0).squeeze(0) for i, window in enumerate(cts)}

    return resampled_cts


def clahe_n_gamma(ct, clip_limit=2.0, tile_grid_size=(8, 8), gamma=0.3, apply_clahe=False, apply_gamma=True, threshold = 0.03):
    """
    Apply CLAHE and gamma correction to a CT scan normalized between 0 and 1.

    Args:
        ct (torch.Tensor): Input tensor of shape (H, W) with values between 0 and 1.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.
        gamma (float): Gamma correction parameter.

    Returns:
        torch.Tensor: Processed image tensor with values between 0 and 1.
    """

    if apply_clahe:
        # Convert the PyTorch tensor to a NumPy array and scale to [0, 255]
        ct_np = (ct.cpu().numpy() * 255).astype(np.uint8)
        # Create a CLAHE object with the desired parameters
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # Apply CLAHE to the image
        clahe_image_np = clahe.apply(ct_np)
        # Convert the processed image back to a PyTorch tensor and scale to [0, 1]
        ct = torch.from_numpy(clahe_image_np.astype(np.float32) / 255.0)
        ct = (ct-ct.min())/(ct.max()-ct.min())
        # Apply threshold to preserve background

    ct[ct < threshold] = 0

    if apply_gamma:
        ct = torch.pow(ct, gamma)

    return ct

def load_n_project_ct(pid, datapath, ct_path,axis=1,save=False,save_path=None,device='cpu'):
    #start=time.time()
    ct,spacing=load_ct(pid, datapath, ct_path, device=device)
    print('Spacing:',spacing)
    print('Shape of CT:',ct.shape)
    #print('time to load:',time.time()-start)
    #start=time.time()
    cts=window_ct(ct)
    #print('time to window:',time.time()-start)
    #start=time.time()
    cts=project_cts(cts, spacing, axis=axis)
    #print('time to project:',time.time()-start)
    #start=time.time()
    cts['skeleton']=clahe_n_gamma(cts['skeleton'],clip_limit=5, tile_grid_size=(8, 8), gamma=0.6, apply_clahe=True, apply_gamma=True, threshold=0.01)
    #print('time to clahe:',time.time()-start)
    #start=time.time()
    if save:
        for window in cts:
            projection=cts[window].unsqueeze(-1).repeat(1,1,3) * 255.0
            # Rotate the projection by 90 degrees counter-clockwise
            projection = torch.rot90(projection, k=1, dims=(0, 1))
            projection_np = projection.detach().cpu().numpy()
            filename = f"{pid}_ct_window_{window}_axis_{axis}.png"
            filepath = os.path.join(save_path, filename) if save_path else filename
            cv2.imwrite(filepath, projection_np)
    #print('time to save:',time.time()-start)

    return cts

def load_mask(pid, organ, datapath, mask_path,device='cuda'):
    # Load the CT scan
    if mask_path is None:
        # Try the 'predictions' path first
        mask_path = os.path.join(datapath, pid, 'predictions', organ + '.nii.gz')
        
        # If the file doesn't exist in 'predictions', check 'segmentations'
        if not os.path.exists(mask_path):
            mask_path = os.path.join(datapath, pid, 'segmentations', organ + '.nii.gz')
    mask_nii = nib.load(mask_path)


    # Calculate the orientation transformation based on the CT scan
    transform = get_orientation_transform(mask_nii, orientation=('L', 'A', 'S')) 

    # Apply the transformation to the CT scan data
    #start=time.time()
    mask = apply_transform(np.asanyarray(mask_nii.dataobj).astype(bool), transform)
    #print('time to reorient:',time.time()-start)

    #start=time.time()
    mask=torch.from_numpy(mask.copy()).float()
    if device!='cpu':
        mask=mask.to(device)
    #print('time to move to device:',time.time()-start)

    return mask

def load_all_masks(pid, datapath, device='cuda',organs=['spleen','stomach','gall_bladder','liver']):
    # Try the first path
    mask_path = os.path.join(datapath, pid, 'segmentations')
    
    # Check if the path exists, if not, use the fallback
    if not os.path.exists(mask_path):
        mask_path = os.path.join(datapath, pid, 'predictions')

    if organs is None:
        organs=[organ[:-len('.nii.gz')] for organ in os.listdir(mask_path)]

    masks=[]
    for pth in organs:
        mask=load_mask(pid, pth, datapath, None,device=device)
        masks.append(mask)
    masks=torch.stack(masks,0)

    return masks,organs



def project_masks(masks, axis=1,th=0.5):
    """
    Projects masks along a given axis and resamples the remaining two dimensions to have 1x1 mm spacing.

    Parameters: 
    masks : torch.Tensor
        A tensor of shape (N, D, H, W) containing the masks for each organ.
    axis : int
        The axis along which to sum the image (default is 1).
    """

    # Sum along the specified axis
    summed_masks = torch.sum(masks, dim=axis+1)#accounts for batching

    # Normalize the masks by dividing by the maximum value of the summed masks
    organ_projection = summed_masks / (summed_masks.amax(dim=(-1, -2), keepdim=True) + 1e-8)  # Normalize

    # Apply threshold if specified
    if th > 0:
        organ_projection = torch.where(organ_projection > 0,
                                 organ_projection / (1 / (1 - th)) + th,
                                 torch.tensor(0.0).type_as(organ_projection))

    return organ_projection


def resize_masks(masks, size):
    masks=F.interpolate(masks, size=size, mode='nearest')
    return masks

def load_n_project_masks(pid, datapath, size=None, device='cuda',axis=1,th=0.5,save=False,save_path=None,organs=None):
    #start=time.time()
    masks,organs=load_all_masks(pid, datapath, device=device,organs=organs)
    #print('time to load:',time.time()-start)
    #start=time.time()
    masks=project_masks(masks, axis=axis,th=th)
    #print('time to project:',time.time()-start)
    #start=time.time()

    if size is not None:
        masks=F.interpolate(masks.unsqueeze(1), size=size, mode='nearest').squeeze(1)

    if save:
        for i,organ in enumerate(organs):
            projection=masks[i] * 255.0

            # Rotate the projection by 90 degrees counter-clockwise
            projection = torch.rot90(projection, k=1, dims=(0, 1))

            projection_np = projection.detach().cpu().numpy()
            filename = f"{pid}_mask_axis_{axis}.png"
            filepath = os.path.join(save_path, filename) if save_path else filename
            cv2.imwrite(filepath, projection_np)
    #print('time to save:',time.time()-start)

    return masks,organs

def overlap_ct_and_masks(cts, masks, organs,device='cpu'):
    """
    Overlay CT scans and masks for different organs and save the resulting images.

    Parameters: 
    cts : dict
        Dictionary of CT scans, where each key represents a window (or scan),
        and the value is a PyTorch tensor representing the CT scan.
    masks : torch.Tensor
        A tensor of shape (N, D, H, W) containing the masks for each organ.
    organs : list
        A list of organ names corresponding to the masks.
    device : str
        The device to run the computations on ('cuda:0' or 'cpu').
    """

    overlays={}
    # Iterate over the CT windows and process each one
    for window in cts:
        # Get the CT scan for the current window
        ct = cts[window]
        ov={}

        # Iterate over the organs and overlay the CT scan with the masks
        for i, organ in enumerate(organs):
            # Get the mask for the current organ
            mask = masks[i]
            mask [mask > 0.5] = 1
            mask [mask <= 0.5] = 0
            mask = mask.bool()
            if 'cuda' in device:
                mask = mask.to(device)

            # Overlay the CT scan with the mask
            overlay=ct.clone().unsqueeze(0).repeat(3,1,1)
            if 'cuda' in device:
                overlay=overlay.to(device)
            overlay[1][mask] = 0.0
            overlay[2][mask] = 0.0

            if window=='skeleton':
                overlay[0][mask] += 0.5
                overlay=torch.clamp(overlay,0,1)

            ov[organ]=overlay
        overlays[window]=ov
    return overlays



def project_ct_and_masks(pid, ct_pth, mask_pth, device='cuda',axis=1,th=0.5,save=False,save_path=None,organs=None):
    if not os.path.exists(os.path.join(save_path, pid,f"{pid}_ct_window_bone_axis_{axis}.png")):
        #start=time.time()
        cts=load_n_project_ct(pid, ct_pth, ct_path=None,axis=axis,save=save,save_path=save_path,device=device)
        #print('time to load and project ct:',time.time()-start)
    else:
        cts={}
        for window in ['skeleton', 'organs','bone']:
            filename = f"{pid}_ct_window_{window}_axis_{axis}.png"
            filepath = os.path.join(save_path, pid, filename) if save_path else filename
            cts[window]=torch.rot90(torch.from_numpy(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)/255.0).float(), k=-1, dims=(0, 1))
            if device!='cpu':
                cts[window]=cts[window].to(device)
        print('ct projection loaded from '+f"{pid}_ct_window_{window}_axis_{axis}.png")

    #start=time.time()
    masks,organs=load_n_project_masks(pid, mask_pth, size=cts['organs'].shape[-2:], device=device,axis=axis,th=th,save=save,save_path=save_path,organs=organs)
    #print('time to load and project masks:',time.time()-start)
    if 'cuda' in device:
        masks=masks.to(device)
    overlay=overlap_ct_and_masks(cts, masks, organs,device=device)
    if save:
        for window in overlay:
            for organ in overlay[window]:
                ov=overlay[window][organ]
                ov *= 255.0
                # Rotate the projection by 90 degrees counter-clockwise
                ov = torch.rot90(ov, k=1, dims=(-2, -1))
                ov = ov.permute(1, 2, 0)
                ov=ov.detach().cpu().numpy()
                ov = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
                filename = pid+'_overlay_window_'+window+'_axis_'+str(axis)+'_'+organ+'.png'
                filepath = os.path.join(save_path, filename) if save_path else filename
                cv2.imwrite(filepath, ov)

    return cts,masks,organs

def project_files_standard(ct_pth, mask_pth, destin, organ, file_list=None, axis=1,device='cpu',skip_existing=True):
    #no multiprocessing
    if file_list is None:
        file_list=[f for f in file_list if f in os.listdir(mask_pth)]
    for pid in file_list:
        os.makedirs(os.path.join(destin,pid), exist_ok=True)
        if skip_existing and os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_bone_axis_'+str(axis)+'_'+organ+'.png')) \
                                            and os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_organs_axis_'+str(axis)+'_'+organ+'.png')) \
                                            and os.path.exists(os.path.join(destin,pid,pid+'_ct_window_bone_axis_'+str(axis)+'.png')) \
                                            and os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_skeleton_axis_'+str(axis)+'_'+organ+'.png')) \
                                            and os.path.exists(os.path.join(destin,pid,pid+'_ct_window_skeleton_axis_'+str(axis)+'.png')):          
            print(f'Skipping {pid}, already exists')
            continue

        print(f'Projecting {pid}')
        start_proj=time.time()
        project_ct_and_masks(pid, ct_pth=ct_pth, mask_pth=mask_pth, device=device,axis=axis,th=0.5,save=True,save_path=os.path.join(destin,pid),organs=[organ])
        print(f'Projected {pid}')
        print('time to project:',time.time()-start_proj)
        print('')

def process_single_file(pid, ct_pth, mask_pth, destin, organ, axis, device, skip_existing):
    os.makedirs(os.path.join(destin, pid), exist_ok=True)
    
    # Check if the necessary files already exist
    #print(f'Checking if file exists in {os.path.join(destin,pid, pid + "_overlay_window_bone_axis_" + str(axis) + "_" + organ + ".png")}')

    if skip_existing and os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_bone_axis_'+str(axis)+'_'+organ+'.png')) \
                     and os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_organs_axis_'+str(axis)+'_'+organ+'.png')) \
                     and os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_skeleton_axis_'+str(axis)+'_'+organ+'.png')) \
                     and os.path.exists(os.path.join(destin,pid,pid+'_ct_window_skeleton_axis_'+str(axis)+'.png')) \
                     and os.path.exists(os.path.join(destin,pid,pid+'_ct_window_organs_axis_'+str(axis)+'.png')) \
                     and os.path.exists(os.path.join(destin,pid,pid+'_ct_window_bone_axis_'+str(axis)+'.png')):
        print(f'Skipping {pid}, already exists')
        return
    print(ct_pth, mask_pth)
    if not os.path.isdir(os.path.join(ct_pth, pid)):
        print(f'Patient {pid} not found')
        return
    
    if 'ct.nii.gz' not in os.listdir(os.path.join(ct_pth, pid)):
        print(f'CT not found for {pid}')
        return
    
    if not os.path.isdir(os.path.join(mask_pth, pid)):
        print(f'Mask {pid} not found')
        return

    # Process the file
    print(f'Projecting {pid}')
    start_proj = time.time()
    
    # Call the function to project CT and masks (assuming this function is defined elsewhere)
    project_ct_and_masks(pid, ct_pth=ct_pth, mask_pth=mask_pth, device=device, axis=axis, th=0.5, save=True, save_path=os.path.join(destin, pid), organs=[organ])
    
    print(f'Projected {mask_pth}/{pid} and saved in {os.path.join(destin, pid)}')
    print('Time to project:', time.time() - start_proj)
    print('')


# Main function that uses multiprocessing to parallelize the task
def project_files(pth=None, destin=None, organ='liver', file_list=None, axis=1, device='cpu', skip_existing=True, num_processes=10, 
                  ct_pth=None, mask_pth=None):
    if ct_pth is None:
        ct_pth = pth
    if mask_pth is None:
        mask_pth = pth

    if 'cuda' in device:
        project_files_standard(ct_pth=ct_pth, mask_pth=mask_pth, destin=destin, organ=organ, 
                               file_list=file_list, axis=axis,device=device,skip_existing=skip_existing)
        return
    
    if file_list is None:
        file_list = [f for f in os.listdir(mask_pth) \
                     if (os.path.isfile(os.path.join(mask_pth, f,'segmentations',organ+'.nii.gz')) \
                         or os.path.isfile(os.path.join(mask_pth, f,'predictions',organ+'.nii.gz')))]  # Load all files in the directory if no list is provided
        file_list = [f for f in file_list if 'ct.nii.gz' in os.listdir(os.path.join(ct_pth, f))]  # Filter out files without a CT scan

    # Create a pool of workers for parallel processing
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Prepare arguments for the helper function
        pool.starmap(
            process_single_file, 
            [(pid, ct_pth, mask_pth, destin, organ, axis, device, skip_existing) for pid in file_list]
        )


def overlay_projection(pid, organ, datapath,save_path,th=0.5,
                       mask_only=False,ct_only=False,
                       clahe=False,he=False,window='organs',
                       ct_path=None, mask_path=None):

    ct, mask=load_ct_and_mask(pid, organ, datapath,
                              ct_path=ct_path, mask_path=mask_path)

    ct_projections=plot_organ_projection_3_axis([ct], organ, pid, th=0,
                                                 ct=True, save=False,
                                                 window=window)
    if clahe or he:
        for i in range(len(ct_projections)):
            img=ct_projections[i]
            flag=False
            #print('max img:',np.max(img))
            #print('min img:',np.min(img))
            if np.max(img)>1:
                img=img/255
                flag=True
            if clahe:
                ct_projections[i] = exposure.equalize_adapthist(img, clip_limit=0.01, 
                                                                nbins=256, 
                                                                kernel_size=(16,16))
            if he:
                ct_projections[i] = exposure.equalize_hist(img)

            if flag:
                ct_projections[i]=ct_projections[i]*255

    mask_projections=plot_organ_projection_3_axis([mask], organ, pid, th=th,
                                                   ct=False, save=False)

    if mask_only:
        ct_projections=[0*proj for proj in ct_projections]
    if ct_only:
        mask_projections=[0*proj for proj in mask_projections]


    for i in range(len(ct_projections)):
        ct_proj=ct_projections[i]
        mask_proj=mask_projections[i]

        print('max ct proj:',np.max(ct_proj))
        print('min ct proj:',np.min(ct_proj))  
        print('max mask proj:',np.max(mask_proj))
        print('min mask proj:',np.min(mask_proj))    

        ct_proj = np.expand_dims(ct_proj, axis=-1)
        ct_proj = np.tile(ct_proj, (1, 1, 3))

        if not mask_only:
            overlay = ct_proj
            ct_proj[:,:,0] = ct_proj[:,:,0]-mask_proj
            ct_proj[:,:,1] = ct_proj[:,:,1]-mask_proj
        else:
            overlay = np.expand_dims(mask_proj, axis=-1)
            overlay = np.tile(overlay, (1, 1, 3))
            overlay[:,:,0] = 0
            overlay[:,:,1] = 0
            

        
        #overlay = overlay.astype(np.uint8)

        print('overlay shape:',overlay.shape)
        print('max overlay:',np.max(overlay))
        print('min overlay:',np.min(overlay))

        print('shape of overlay:',overlay.shape)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name='_overlay'
        if mask_only:
            name='_mask'
        if ct_only:
            name='_ct'
        cv2.imwrite(os.path.join(save_path, pid + name + '_axis_'+str(i)+'.png'), 
                    overlay)

        print('Organ projection of ' + save_path + ' for patient ' + pid + \
            ' is saved to ' + os.path.join(save_path, pid + '.png'))
        
def apply_clahe_to_tensor(image_tensor, clip_limit=2.0, tile_grid_size=(8, 8),apply_erosion=True,erosion_kernel_size=9):
    """
    Apply CLAHE to a PyTorch tensor image normalized between 0 and 1.

    Args:
        image_tensor (torch.Tensor): Input tensor of shape (C, H, W) or (H, W) with values between 0 and 1.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.

    Returns:
        torch.Tensor: Processed image tensor with values between 0 and 1.
    """

    # Ensure the image is a 2D grayscale tensor (H, W) or (1, H, W)
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.squeeze(0)  # Remove channel dimension if it exists
    
    # Convert the PyTorch tensor to a NumPy array and scale to [0, 255]
    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

    # Create a CLAHE object with the desired parameters
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the image
    clahe_image_np = clahe.apply(image_np)

    # Convert the processed image back to a PyTorch tensor and scale to [0, 1]
    clahe_image_tensor = torch.from_numpy(clahe_image_np.astype(np.float32) / 255.0)

    # Add back the channel dimension if it was initially present
    if len(image_tensor.shape) == 2:  # if the original tensor was 3D
        clahe_image_tensor = clahe_image_tensor.unsqueeze(0)

    if apply_erosion:
        _, binary_image = cv2.threshold(clahe_image_np, 0, 1, cv2.THRESH_BINARY)
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)  # Create a square kernel
        eroded_image = cv2.erode(binary_image, kernel, iterations=1)
        binary_image = eroded_image
        binary_tensor = torch.tensor(binary_image)
        if len(image_tensor.shape) == 2:  # if the original tensor was 3D
            binary_tensor = binary_tensor.unsqueeze(0)
        clahe_image_tensor = clahe_image_tensor * binary_tensor


    return clahe_image_tensor

def plot_organ_projection_cuda(list_of_array, organ_name, pid, axis=1,
                               pngpath=None, th=0.5, ct=False, save=True,
                               window='organs'):
    # Stack tensors along a new dimension
    stacked_x = torch.stack(list_of_array, dim=0)  # Shape: (N, D1, D2, D3)

    # Apply windowing if ct is True
    if ct:
        if window == 'organs':
            upper_limit = 250.0
            lower_limit = -150.0
            offset = 150.0
            divisor = 400.0
        elif window == 'bone':
            upper_limit = 1500.0
            lower_limit = -500.0
            offset = 500.0
            divisor = 2000.0
        elif window == 'skeleton':
            upper_limit = 2000.0
            lower_limit = 400.0
            offset = -400.0
            divisor = 1600.0
        else:
            raise ValueError('Window should be "organs" "skeleton" or "bone"')

        # Apply the windowing to the stacked tensor
        stacked_x = torch.clamp(stacked_x, min=lower_limit, max=upper_limit)
        stacked_x = (stacked_x + offset) / divisor


    # Sum over the specified axis (adjusted for batch dimension)
    organ_projection = torch.sum(stacked_x, dim=axis+1)

    # Normalize each projection in the batch
    max_vals = organ_projection.view(organ_projection.size(0), -1).max(dim=1)[0] + 1e-8
    organ_projection = organ_projection / max_vals[:, None, None]

    # Sum the normalized projections across the batch dimension
    projection = torch.sum(organ_projection, dim=0)

    # Normalize the final projection
    projection = projection / (torch.max(projection) + 1e-8)

    # Apply threshold if specified
    if th > 0:
        projection = torch.where(projection > 0,
                                 projection / (1 / (1 - th)) + th,
                                 torch.tensor(0.0).type_as(projection))
        
    
    if window == 'skeleton' and ct:
        #contrast enhancement
        #projection[projection > 0]+=0.05
        projection=apply_clahe_to_tensor(projection.unsqueeze(0),clip_limit=5,apply_erosion=False).squeeze(0)
        gamma=0.3
        projection = torch.pow(projection, gamma)
        projection = (projection-projection.min())/(projection.max()-projection.min())
        # Apply threshold to preserve background
        threshold = 0.03  # Adjust as needed
        projection[projection < threshold] = 0
        projection = torch.clamp(projection, min=0, max=1)

    # Scale to 255 for image representation
    projection *= 255.0

    # Rotate the projection by 90 degrees counter-clockwise
    projection = torch.rot90(projection, k=1, dims=(0, 1))

    # Save the projection if required
    if save:
        projection_np = projection.detach().cpu().numpy()
        if pngpath is not None and not os.path.exists(pngpath):
            os.makedirs(pngpath)
        filename = f"{pid}_axis_{axis}.png"
        filepath = os.path.join(pngpath, filename) if pngpath else filename
        cv2.imwrite(filepath, projection_np)
        print(f'Organ projection of {organ_name} for patient {pid} is saved to {filepath}')

    return projection



def overlay_projection_fast(pid, organ, datapath, save_path, th=0.5,
                            mask_only=False, ct_only=False, window='organs',
                            ct_path=None, mask_path=None, axis=1, device='cuda:0',
                            precision=32):
    """
    Generate and save overlay projections of CT and mask images.

    Parameters:
    - pid (str): Patient ID.
    - organ (str): Organ name.
    - datapath (str): Path to the data directory.
    - save_path (str): Directory to save the overlay images.
    - th (float): Threshold for mask projection.
    - mask_only (bool): If True, only the mask is projected.
    - ct_only (bool): If True, only the CT is projected.
    - window (str): Windowing option for CT images ('organs' or 'bone').
    - ct_path (str): Path to the CT image file.
    - mask_path (str): Path to the mask image file.
    - axis (int): Axis along which to project (0, 1, or 2).
    - device (str): Device to run the computations on ('cuda:0' or 'cpu').

    Returns:
    - None
    """
    print(f'Projecting {organ} for patient {pid} from {datapath}')
    # Load CT and mask images
    ct, mask = load_ct_and_mask(pid, organ, datapath,
                                ct_path=ct_path, mask_path=mask_path)

    # Convert to torch tensors and move to device
    # Ensure arrays have positive strides by making a copy
    ct = ct.copy()
    mask = mask.copy()

    ct = torch.from_numpy(ct).to(device).float()
    mask = torch.from_numpy(mask).to(device).float()
    

    # Generate projections
    if not mask_only:
        ct_projection = plot_organ_projection_cuda([ct], organ, pid, axis=axis,
                                                   th=0, ct=True, save=False,
                                                   window=window)
    if not ct_only:
        mask_projection = plot_organ_projection_cuda([mask], organ, pid, axis=axis,
                                                     th=th, ct=False, save=False)

    # Handle cases where only CT or only mask is desired
    if mask_only:
        # Create a zero projection for CT
        ct_projection = torch.zeros_like(mask_projection, device=device)
    if ct_only:
        # Create a zero projection for mask
        mask_projection = torch.zeros_like(ct_projection, device=device)

    # Ensure projections are on the same device and have the same shape
    ct_projection = ct_projection.to(device)
    mask_projection = mask_projection.to(device)

    # Prepare overlay
    if mask_only:
        # Overlay only the mask in the blue channel
        overlay = torch.stack([
            torch.zeros_like(mask_projection),  # R channel
            torch.zeros_like(mask_projection),  # G channel
            mask_projection  # B channel
        ], dim=2)
    elif ct_only:
        # Overlay only the CT in all channels
        overlay = ct_projection.unsqueeze(-1).repeat(1, 1, 3)
    else:
        if window!='skeleton':
            overlay = torch.stack([
            ct_projection - mask_projection,  # B channel
            ct_projection - mask_projection,  # G channel
            ct_projection  # R channel
        ], dim=2)
            
        else:
            overlay = torch.stack([
                ct_projection - mask_projection,  # B channel
                ct_projection - mask_projection,  # G channel
                ct_projection + (mask_projection*0.8)  # R channel
            ], dim=2)


    # Clamp values to [0, 255] and convert to uint8
    overlay = torch.clamp(overlay, 0, 255).byte()

    # Move to CPU and convert to NumPy array
    overlay_np = overlay.cpu().numpy()

    # Print overlay information
    #print('overlay shape:', overlay_np.shape)
    #print('max overlay:', overlay_np.max())
    #print('min overlay:', overlay_np.min())

    # Save the overlay image
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    name = '_overlay'
    if mask_only:
        name = '_mask'
    if ct_only:
        name = '_ct'
    filename = os.path.join(save_path, f"{pid}{name}_window_{window}_axis_{axis}_{organ}.png")
    cv2.imwrite(filename, overlay_np)
    print(f'Organ projection of {organ} for patient {pid} is saved to {filename}')

def create_composite_image(pth, organ, axis=1, y1_bone=None, y2_bone=None, y1_organs=None, y2_organs=None,name=''):

    if y1_bone is None:
        y1_bone = os.path.join(pth, f'y1_bone_overlay_window_bone_axis_{axis}_{organ}.png')
        y1_organs = os.path.join(pth, f'y1_organs_overlay_window_organs_axis_{axis}_{organ}.png')
        y2_bone = os.path.join(pth, f'y2_bone_overlay_window_bone_axis_{axis}_{organ}.png')
        y2_organs = os.path.join(pth, f'y2_organs_overlay_window_organs_axis_{axis}_{organ}.png')

    # Load the images
    image_paths = [y1_bone, y1_organs, y2_bone, y2_organs]
    images = [Image.open(path) for path in image_paths]

    # Get image dimensions in pixels
    img_width, img_height = images[0].size

    # DPI (dots per inch)
    dpi = 100  # Adjust as needed

    # Calculate subplot size in inches
    subplot_width = img_width / dpi
    subplot_height = img_height / dpi

    # Title font size in points and inches
    title_font_size_pts = 12  # Default matplotlib font size
    title_font_size_inch = title_font_size_pts / 72  # Convert points to inches

    # Desired spacing between subplots (1.5 times title font size)
    desired_spacing_inch = 1.5 * title_font_size_inch

    # Compute wspace and hspace as fractions of subplot size
    wspace = desired_spacing_inch / subplot_width
    hspace = desired_spacing_inch / subplot_height

    # Total figure size in inches
    fig_width = 2 * subplot_width + desired_spacing_inch
    fig_height = 2 * subplot_height + desired_spacing_inch

    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=dpi)
    axes = axes.flatten()

    # Use your original titles
    titles = ["Image 1 - Overlay 1", "Image 2 - Overlay 1", "Image 3 - Overlay 2", "Image 4 - Overlay 2"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

        # Add 'R' and 'L' labels with some padding
        padding = 10  # Adjust as needed
        ax.text(padding, img_height / 2, 'R', fontsize=title_font_size_pts, ha='left', va='center',
                color='blue', fontweight='bold')
        ax.text(img_width - padding, img_height / 2, 'L', fontsize=title_font_size_pts, ha='right', va='center',
                color='green', fontweight='bold')

    # Adjust subplot spacing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=wspace, hspace=hspace)

    # Save the figure
    save_path = os.path.join(pth, name+f'composite_image_axis_{axis}_{organ}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    print(f'Composite image saved to {save_path}')
    plt.close()
    
def create_composite_image_2figs(pth, organ, axis=1, y1_bone=None, y2_bone=None,name='',window='bone',
                                 just_ct_name=False):

    if y1_bone is None:
        y1_bone = os.path.join(pth, f'y1_bone_overlay_window_{window}_axis_{axis}_{organ}.png')
        y2_bone = os.path.join(pth, f'y2_bone_overlay_window_{window}_axis_{axis}_{organ}.png')

    # Load the images
    image_paths = [y1_bone, y2_bone]
    images = [Image.open(path).convert('RGB') for path in image_paths]

    # Get image dimensions in pixels
    img_width, img_height = images[0].size

    # DPI (dots per inch)
    dpi = 100  # Adjust as needed

    # Calculate subplot size in inches
    subplot_width = img_width / dpi
    subplot_height = img_height / dpi

    # Title font size in points and inches
    title_font_size_pts = 12  # Default matplotlib font size
    title_font_size_inch = title_font_size_pts / 72  # Convert points to inches

    # Desired spacing between subplots (1.5 times title font size)
    desired_spacing_inch = 1.5 * title_font_size_inch

    # Compute wspace as a fraction of subplot width
    wspace = desired_spacing_inch / subplot_width

    # Total figure size in inches
    fig_width = 2 * subplot_width + desired_spacing_inch
    fig_height = subplot_height + desired_spacing_inch  # Adjust height if needed

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    axes = axes.flatten()

    # Titles for each subplot
    titles = ["Image 1", "Image 2"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

        # Add 'R' and 'L' labels with some padding
        padding = 10  # Adjust as needed
        ax.text(padding, img_height / 2, 'R', fontsize=title_font_size_pts, ha='left', va='center',
                color='blue', fontweight='bold')
        ax.text(img_width - padding, img_height / 2, 'L', fontsize=title_font_size_pts, ha='right', va='center',
                color='green', fontweight='bold')

    # Adjust subplot spacing
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05, wspace=wspace)

    # Save the figure
    if just_ct_name:
        if window=='skeleton':
            save_path = os.path.join(pth, name+f'composite_ct_2_figs_axis_{axis}_skeleton.png')
        else:
            save_path = os.path.join(pth, name+f'composite_ct_2_figs_axis_{axis}.png')
    else:
        if window=='skeleton':
            save_path = os.path.join(pth, name+f'composite_image_2_figs_axis_{axis}_{organ}_skeleton.png')
        else:
            save_path = os.path.join(pth, name+f'composite_image_2_figs_axis_{axis}_{organ}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    print(f'Composite image saved to {save_path}')
    plt.close()

# to improve speed: project ct just once. Then you project the masks using lower precision and overlay them. Use torch for resampling (current function has bug)

def composite_dataset_liver(output_dir='projections',path='projections_bad_liver_overlay_bone/', axis=1, organ='liver',
                            file_list=None):
    os.makedirs(output_dir, exist_ok=True)
    if file_list is None:
        file_list=os.listdir(path)

    for file in file_list:
        if 'overlay_axis_'+str(axis) not in file:
            continue
        ct='projections_bad_liver_overlay_bone/'+file.replace('overlay_','ct_')
        y1_bone='projections_bad_liver_overlay_bone/'+file
        y1_organs='projections_bad_liver_overlay/'+file
        y2_bone='projections_improved_liver_overlay_bone/'+file
        y2_organs='projections_improved_liver_overlay/'+file

        shutil.copy(ct, os.path.join(output_dir,file.split('.')[0]+'_ct_axis_1_liver.png'))
        create_composite_image(output_dir, organ, axis, y1_bone=y1_bone, y2_bone=y2_bone, y1_organs=y1_organs, y2_organs=y2_organs,name=file.split('.')[0]+'_')
        create_composite_image_2figs(output_dir, organ, axis, y1_bone=y1_bone, y2_bone=y2_bone,name=file.split('.')[0]+'_')


def project_files_slow(pth, destin, organ, file_list, axis=1,device='cuda:0',skip_existing=True):
    file_list=[f for f in file_list if f in os.listdir(pth)]
    for pid in file_list:
        os.makedirs(os.path.join(destin,pid), exist_ok=True)
        if skip_existing and os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_bone_axis_'+str(axis)+'_'+organ+'.png')) \
                                            and os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_organs_axis_'+str(axis)+'_'+organ+'.png')) \
                                            and os.path.exists(os.path.join(destin,pid,pid+'_ct_window_bone_axis_'+str(axis)+'.png')) \
                                            and os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_skeleton_axis_'+str(axis)+'_'+organ+'.png')) \
                                            and os.path.exists(os.path.join(destin,pid,pid+'_ct_window_skeleton_axis_'+str(axis)+'.png')):          
            print(f'Skipping {pid}, already exists')
            continue
        
        if not os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_organs_axis_'+str(axis)+'_'+organ+'.png')):
            overlay_projection_fast(pid=pid, organ=organ, datapath=pth, 
                                    save_path=os.path.join(destin,pid), 
                                    th=0.5,mask_only=False, ct_only=False, window='organs',
                                    ct_path=None, mask_path=None, axis=1, device=device,
                                    precision=32)
        
        if not os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_bone_axis_'+str(axis)+'_'+organ+'.png')):
            overlay_projection_fast(pid=pid, organ=organ, datapath=pth, 
                                    save_path=os.path.join(destin,pid), 
                                    th=0.5,mask_only=False, ct_only=False, window='bone',
                                    ct_path=None, mask_path=None, axis=1, device=device,
                                    precision=32)
        
        if not os.path.exists(os.path.join(destin,pid,pid+'_ct_window_bone_axis_'+str(axis)+'_'+organ+'.png')):
            overlay_projection_fast(pid=pid, organ=organ, datapath=pth, 
                                    save_path=os.path.join(destin,pid), 
                                    th=0.5,mask_only=False, ct_only=True, window='bone',
                                    ct_path=None, mask_path=None, axis=1, device=device,
                                    precision=32)
        
        if not os.path.exists(os.path.join(destin,pid,pid+'_ct_window_skeleton_axis_'+str(axis)+'_'+organ+'.png')):
            overlay_projection_fast(pid=pid, organ=organ, datapath=pth, 
                                    save_path=os.path.join(destin,pid), 
                                    th=0.5,mask_only=False, ct_only=True, window='skeleton',
                                    ct_path=None, mask_path=None, axis=1, device=device,
                                    precision=32)
        
        if not os.path.exists(os.path.join(destin,pid,pid+'_overlay_window_skeleton_axis_'+str(axis)+'_'+organ+'.png')):
            overlay_projection_fast(pid=pid, organ=organ, datapath=pth, 
                                    save_path=os.path.join(destin,pid), 
                                    th=0.5,mask_only=False, ct_only=False, window='skeleton',
                                    ct_path=None, mask_path=None, axis=1, device=device,
                                    precision=32)
        
def composite_dataset(output_dir, good_path, bad_path, axis=1,organ=None,fast=False,
                            file_list=None):
    path1=bad_path
    path2=good_path
    if organ is None:
        organs=os.listdir(path1)
    else:
        organs=[organ]
    for organ in organs:
        if organ=='all_classes':
            continue
        os.makedirs(os.path.join(output_dir,organ), exist_ok=True)
        if file_list is None:
            lst=os.listdir(os.path.join(path1,organ))
        else:
            lst=file_list[organ]
        for file in lst:
            if file not in os.listdir(os.path.join(path2,organ)):
                print(f'File {file} does not exist in {path2},skipping')
                #raise ValueError(f'File {file} does not exist in {path2},skipping')
                continue
            
            ct=os.path.join(path1,organ,file,file+'_ct_window_bone_axis_'+str(axis)+'.png')
            y1_bone=os.path.join(path1,organ,file,file+'_overlay_window_bone_axis_'+str(axis)+'_'+organ+'.png')
            y1_organs=os.path.join(path1,organ,file,file+'_overlay_window_organs_axis_'+str(axis)+'_'+organ+'.png')
            y2_bone=os.path.join(path2,organ,file,file+'_overlay_window_bone_axis_'+str(axis)+'_'+organ+'.png')
            y2_organs=os.path.join(path2,organ,file,file+'_overlay_window_organs_axis_'+str(axis)+'_'+organ+'.png')
            skeleton=os.path.join(path1,organ,file,file+'_ct_window_skeleton_axis_'+str(axis)+'.png')
            y1_seleton=os.path.join(path1,organ,file,file+'_overlay_window_skeleton_axis_'+str(axis)+'_'+organ+'.png')
            y2_seleton=os.path.join(path2,organ,file,file+'_overlay_window_skeleton_axis_'+str(axis)+'_'+organ+'.png')

            #if os.path.isfile(os.path.join(output_dir,organ,file.split('.')[0]+'_overlay_window_skeleton_axis_'+str(axis)+'_'+organ+'_y2.png')):
            #    print(f'Skipping {file}, already exists')
            #    continue

            if not os.path.exists(ct):
                print(f'File {file} does not exist in {path2}')
                #raise ValueError(f'File {file} does not exist in {path2},skipping')
                continue
            
            shutil.copy(ct, os.path.join(output_dir,organ,file.split('.')[0]+'_ct_window_bone_axis_'+str(axis)+'.png'))
            shutil.copy(y1_bone, os.path.join(output_dir,organ,file.split('.')[0]+'_overlay_window_bone_axis_'+str(axis)+'_'+organ+'_y1.png'))
            shutil.copy(y1_organs, os.path.join(output_dir,organ,file.split('.')[0]+'_overlay_window_organs_axis_'+str(axis)+'_'+organ+'_y1.png'))
            shutil.copy(y2_bone, os.path.join(output_dir,organ,file.split('.')[0]+'_overlay_window_bone_axis_'+str(axis)+'_'+organ+'_y2.png'))
            shutil.copy(y2_organs, os.path.join(output_dir,organ,file.split('.')[0]+'_overlay_window_organs_axis_'+str(axis)+'_'+organ+'_y2.png'))
            shutil.copy(skeleton, os.path.join(output_dir,organ,file.split('.')[0]+'_ct_window_skeleton_axis_'+str(axis)+'.png'))
            shutil.copy(y1_seleton, os.path.join(output_dir,organ,file.split('.')[0]+'_overlay_window_skeleton_axis_'+str(axis)+'_'+organ+'_y1.png'))
            shutil.copy(y2_seleton, os.path.join(output_dir,organ,file.split('.')[0]+'_overlay_window_skeleton_axis_'+str(axis)+'_'+organ+'_y2.png'))
            print('Copied files: ', os.path.join(output_dir,organ,file.split('.')[0]+'_ct_window_bone_axis_'+str(axis)+'.png'))

            if not fast:
                create_composite_image(os.path.join(output_dir,organ), organ, axis, y1_bone=y1_bone, y2_bone=y2_bone, y1_organs=y1_organs, y2_organs=y2_organs,name=file.split('.')[0]+'_')
                create_composite_image_2figs(os.path.join(output_dir,organ), organ, axis, y1_bone=y1_bone, y2_bone=y2_bone,name=file.split('.')[0]+'_')
                create_composite_image_2figs(os.path.join(output_dir,organ), organ, axis, y1_bone=y1_seleton, y2_bone=y2_seleton,name=file.split('.')[0]+'_',window='skeleton')
                create_composite_image_2figs(os.path.join(output_dir,organ), organ, axis, y1_bone=ct, y2_bone=skeleton,name=file.split('.')[0]+'_',window='skeleton',just_ct_name=True)
                create_composite_image_2figs(os.path.join(output_dir,organ), organ, axis, y1_bone=y2_bone, y2_bone=y1_bone,name=file.split('.')[0]+'_best1_')#invert y1 and y2
                highlight_skeleton(ct_path=ct, skeleton_path=skeleton, pth=os.path.join(output_dir,organ),name=file.split('.')[0]+'_',red=True)
                highlight_skeleton(ct_path=ct, skeleton_path=skeleton, pth=os.path.join(output_dir,organ),name=file.split('.')[0]+'_',red=False)


def join_left_and_right_colorful(image_path1, image_path2):
    # Load images using PIL
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Convert images to numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    print(image1_array.shape, image2_array.shape)

    # Check if images have 4 channels (RGBA)
    if image1_array.shape[-1] == 4:
        image1_array = image1_array[:, :, :3]  # Drop alpha channel if present
    if image2_array.shape[-1] == 4:
        image2_array = image2_array[:, :, :3]  # Drop alpha channel if present

    # Ensure the images have 3 channels (RGB)
    if image1_array.shape[-1] != 3 or image2_array.shape[-1] != 3:
        raise ValueError("Both images must be RGB with 3 channels.")

    # Get red, green, and blue channels from image 1 and image 2
    red1 = image1_array[:, :, 0]  # Red channel of image 1
    green1 = image1_array[:, :, 1]  # Green channel of image 1
    blue1 = image1_array[:, :, 2]   # Blue channel of image 1

    red2 = image2_array[:, :, 0]  # Red channel of image 2
    green2 = image2_array[:, :, 1]  # Green channel of image 2
    blue2 = image2_array[:, :, 2]   # Blue channel of image 2

    # Create a mask where green and blue channels in image 1 are zero
    mask1 = (red1 != green1) & (red1 != blue1)
    mask2 = (red2 != green2) & (red2 != blue2)
    overlap = mask1 & mask2

    #remove red overlay from image 2
    image2_array[mask2,1] = image2_array[mask2,0]
    image2_array[mask2,2] = image2_array[mask2,0]

    grey=image2_array.copy()

    #Make right kidney blue (mask1: set red and green to 0, keep blue)
    image2_array[mask1, 0] = 0  # Set red to 0
    image2_array[mask1, 1] = 0  # Set green to 0
    image2_array[mask1, 2] = grey[mask1, 2] 
    # Keep blue channel as is (preserving the details in the blue channel)

    # Make left kidney green (mask2: set red and blue to 0, keep green)
    image2_array[mask2, 0] = 0  # Set red to 0
    image2_array[mask2, 1] = grey[mask2, 1]  # Set blue to 0
    image2_array[mask2, 2] = 0  # Set blue to 0
    # Keep green channel as is (preserving the details in the green channel)

    # Make overlap red (overlap mask: set green and blue to 0, keep red)
    image2_array[overlap, 0] = grey[overlap, 0]  # Set red to 0
    image2_array[overlap, 1] = 0    # Set green to 0
    image2_array[overlap, 2] = 0    # Set blue to 0
    # Keep red channel as is (preserving the details in the red channel)

    # Convert the modified array back to an image
    result_image = Image.fromarray(image2_array)

    return result_image

def join_left_and_right(image_path1, image_path2):
    # Load images using PIL
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Convert images to numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    print(image1_array.shape, image2_array.shape)

    # Check if images have 4 channels (RGBA)
    if image1_array.shape[-1] == 4:
        image1_array = image1_array[:, :, :3]  # Drop alpha channel if present
    if image2_array.shape[-1] == 4:
        image2_array = image2_array[:, :, :3]  # Drop alpha channel if present

    # Ensure the images have 3 channels (RGB)
    if image1_array.shape[-1] != 3 or image2_array.shape[-1] != 3:
        print(image1_array.shape, image2_array.shape)
        raise ValueError("Both images must be RGB with 3 channels.")

    # Get red, green, and blue channels from image 1 and image 2
    red1 = image1_array[:, :, 0]  # Red channel of image 1
    green1 = image1_array[:, :, 1]  # Green channel of image 1
    blue1 = image1_array[:, :, 2]   # Blue channel of image 1

    red2 = image2_array[:, :, 0]  # Red channel of image 2
    green2 = image2_array[:, :, 1]  # Green channel of image 2
    blue2 = image2_array[:, :, 2]   # Blue channel of image 2

    # Create a mask where green and blue channels in image 1 are zero
    mask1 = (red1 != green1) & (red1 != blue1)
    mask2 = (red2 != green2) & (red2 != blue2)
    overlap = mask1 & mask2

    #remove red overlay from image 2
    image2_array[mask2,1] = image2_array[mask2,0]
    image2_array[mask2,2] = image2_array[mask2,0]

    grey=image2_array.copy()

    #Make right kidney blue (mask1: set red and green to 0, keep blue)
    image2_array[mask1, 0] = grey[mask1, 0]  # Set red to 0
    image2_array[mask1, 1] = 0  # Set green to 0
    image2_array[mask1, 2] = 0 
    # Keep blue channel as is (preserving the details in the blue channel)

    # Make left kidney green (mask2: set red and blue to 0, keep green)
    image2_array[mask2, 0] = grey[mask2, 0]  # Set red to 0
    image2_array[mask2, 1] = 0  # Set blue to 0
    image2_array[mask2, 2] = 0  # Set blue to 0
    # Keep green channel as is (preserving the details in the green channel)

    # Make overlap red (overlap mask: set green and blue to 0, keep red)
    image2_array[overlap, 0] = grey[overlap, 0]  # Set red to 0
    image2_array[overlap, 1] = 0    # Set green to 0
    image2_array[overlap, 2] = 0    # Set blue to 0
    # Keep red channel as is (preserving the details in the red channel)

    # Convert the modified array back to an image
    result_image = Image.fromarray(image2_array)

    return result_image


def highlight_skeleton(ct_path, skeleton_path, pth, name, device='cuda:0',red=True):
    # Load images using PIL
    ct = Image.open(ct_path)
    skeleton = Image.open(skeleton_path)

    # Convert images to numpy arrays
    ct = torch.from_numpy(np.array(ct)).to(device).float()
    skeleton = torch.from_numpy(np.array(skeleton)).to(device).float()


    # Check if images have 4 channels (RGBA)
    if ct.shape[-1] == 4:
        ct = ct[:, :, :3]  # Drop alpha channel if present
    if skeleton.shape[-1] == 4:
        skeleton = skeleton[:, :, :3]  # Drop alpha channel if present

    # Ensure the images have 3 channels (RGB)
    if len(ct.shape) == 2:
        ct = ct.unsqueeze(-1).repeat(1, 1, 3)
        skeleton = skeleton.unsqueeze(-1).repeat(1, 1, 3)

    if ct.shape[-1] != 3 or skeleton.shape[-1] != 3:
        print(ct.shape, skeleton.shape)
        raise ValueError("Both images must be RGB with 3 channels.")

    if red:
        red_skeleton = skeleton.clone()
        red_skeleton[:,:,1] = red_skeleton[:,:,1]+0.3*red_skeleton.max()
        red_skeleton[:,:,1] = 0
        red_skeleton[:,:,2] = 0
        red_ct = ct.clone()
        red_ct[:,:,1] = red_ct[:,:,1]
        red_ct[:,:,1] = 0
        red_ct[:,:,2] = 0
        red_skeleton = torch.clamp(red_skeleton,0,255)
        # Get red, green, and blue channels from image 1 and image 2
        highlighted=torch.where(skeleton>0.0,red_ct,ct)
    else:
        #increase gamma over bones
        ct=ct/255.0
        skeleton=skeleton/255.0
        gamma=0.5
        highlighted=torch.where(skeleton>0,ct*1.5,ct)
        highlighted=torch.clamp(highlighted,0,1)
        highlighted=highlighted*255.0

    highlighted=highlighted.cpu().numpy().astype(np.uint8)
    highlighted=Image.fromarray(highlighted)
    if red:
        save_path = os.path.join(pth, name+f'highlighted_skeleton_red.png')
    else:
        save_path = os.path.join(pth, name+f'highlighted_skeleton.png')
    highlighted.save(save_path)
    print(f'Highlighted skeleton saved to {save_path}')

def join_left_and_right_dataset(folder1, folder2, destination):
    if 'right' not in folder1:
        tmp=folder1
        folder1=folder2
        folder2=tmp

    print(folder1, folder2)

    os.makedirs(destination, exist_ok=True)

    # Get the list of files in folder1
    files_in_folder1 = os.listdir(folder1)

    for file_name in files_in_folder1:
        if '_ct_' in file_name:
            shutil.copy(os.path.join(folder1, file_name), os.path.join(destination, file_name.replace('kidney_right', 'kidneys')))
            continue
        
        # Construct the full paths for the images in folder1 and folder2
        image1_path = os.path.join(folder1, file_name)
        image2_path = os.path.join(folder2, file_name.replace('right', 'left'))

        if not os.path.isfile(image2_path):
            print(f"File {image2_path} does not exist, skipping")
            continue

        print(image1_path, image2_path)

        # Process the images using the previously defined function
        result_image = join_left_and_right(image1_path, image2_path)

        # Save the result to folder3
        result_image_path = os.path.join(destination, file_name)
        result_image.save(result_image_path.replace('kidney_right', 'kidneys'))
        print(f"Processed and saved: {result_image_path.replace('kidney_right', 'kidneys')}")