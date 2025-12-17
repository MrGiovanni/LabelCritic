import os
import argparse
import ast
import importlib
import json
import nibabel as nib

import numpy as np
try:
    import AnnotationVLM.Projection as pj
except:
    import projection as pj
importlib.reload(pj)



def parse_organs_arg(organs_str):
    # Check if the string starts with '[' and ends with ']'
    if organs_str.startswith('[') and organs_str.endswith(']'):
        # Remove the square brackets and split the string by commas
        organs_list = organs_str[1:-1].split(',')
        # Strip any extra spaces from the organ names
        organs_list = [organ.strip() for organ in organs_list]
        return organs_list
    # Return as a single element list if it's not a list format
    return None

def extract_organs_from_combined_mask(mask_path):
    img = nib.load(mask_path)
    data = img.get_fdata()
    present_labels = np.unique(data).astype(int)
    return [organ for organ, idx in LABELS.items() if idx in present_labels and organ != "background"]

def main():
    import tempfile
    import shutil
    import nibabel as nib
    import numpy as np

    parser = argparse.ArgumentParser(description='Project and compare two CT + mask pairs.')
    parser.add_argument('--ct_good', required=True, help='Path to single good CT file')
    parser.add_argument('--mask_good', required=True, help='Path to single good mask file or folder')
    parser.add_argument('--ct_bad', required=True, help='Path to single bad CT file')
    parser.add_argument('--mask_bad', required=True, help='Path to single bad mask file or folder')
    parser.add_argument('--output_dir', required=True, help='Directory to save the comparison result')
    parser.add_argument('--organ', required=True, help='Directory to save the comparison result')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_processes', default='4')
    parser.add_argument('--axis', default=1, type=int)

    args = parser.parse_args()
    
    if not args.organ.endswith('.nii.gz'):
        args.organ += '.nii.gz'

    is_combined = args.mask_good.endswith('.nii.gz') and args.mask_bad.endswith('.nii.gz')

    # Create temporary folders
    temp_ct_good = tempfile.mkdtemp(prefix='ct_good_')
    temp_ct_bad = tempfile.mkdtemp(prefix='ct_bad_')
    temp_mask_good = tempfile.mkdtemp(prefix='mask_good_')
    temp_mask_bad = tempfile.mkdtemp(prefix='mask_bad_')

    try:
        # Create case001 folder and move CTs
        ct_good_case_dir = os.path.join(temp_ct_good, 'case001')
        ct_bad_case_dir = os.path.join(temp_ct_bad, 'case001')
        os.makedirs(ct_good_case_dir, exist_ok=True)
        os.makedirs(ct_bad_case_dir, exist_ok=True)
        shutil.copy(args.ct_good, os.path.join(ct_good_case_dir, 'ct.nii.gz'))
        shutil.copy(args.ct_bad, os.path.join(ct_bad_case_dir, 'ct.nii.gz'))

        
        
        #organs = os.listdir(args.mask_good)
        organs = [args.organ]
        file_list = {organ.replace(".nii.gz", ""): ['case001'] for organ in organs}

        mask_good_case_dir = os.path.join(temp_mask_good, 'case001' ,'segmentations')
        mask_bad_case_dir = os.path.join(temp_mask_bad, 'case001', 'segmentations')
        os.makedirs(mask_good_case_dir, exist_ok=True)
        os.makedirs(mask_bad_case_dir, exist_ok=True)

        for organ in organs:
            shutil.copy(os.path.join(args.mask_good, organ), os.path.join(mask_good_case_dir,  organ))
            shutil.copy(os.path.join(args.mask_bad, organ), os.path.join(mask_bad_case_dir, organ))

            good_mask_path = os.path.join(mask_good_case_dir, organ)
            bad_mask_path = os.path.join(mask_bad_case_dir, organ)

            good_proj = os.path.join(args.output_dir, 'good_projection', organ.replace(".nii.gz", ""))
            bad_proj = os.path.join(args.output_dir, 'bad_projection', organ.replace(".nii.gz", ""))
            os.makedirs(good_proj, exist_ok=True)
            os.makedirs(bad_proj, exist_ok=True)

            pj.project_files(
                    ct_pth=temp_ct_good,
                    mask_pth=temp_mask_good,
                    destin=good_proj,
                    file_list=['case001'],
                    organ=organ.replace(".nii.gz", ""),
                    device=args.device,
                    num_processes=int(args.num_processes),
                    skip_existing=True,
                    axis=args.axis
            )

            pj.project_files(
                    ct_pth=temp_ct_bad,
                    mask_pth=temp_mask_bad,
                    destin=bad_proj,
                    file_list=['case001'],
                    organ=organ.replace(".nii.gz", ""),
                    device=args.device,
                    num_processes=int(args.num_processes),
                    skip_existing=True,
                    axis=args.axis
            )

            pj.composite_dataset(
                    output_dir=args.output_dir,
                    good_path=os.path.join(args.output_dir, 'good_projection'),
                    bad_path=os.path.join(args.output_dir, 'bad_projection'),
                    organ=organ.replace(".nii.gz", ""),
                    fast=False,
                    file_list=file_list,
                    axis=args.axis
            )
        for organ in organs:
            organ = organ.replace(".nii.gz", "")
            if 'left' in organ:
                pj.join_left_and_right_dataset(
                    os.path.join(args.output_dir, organ),
                    os.path.join(args.output_dir, organ.replace('left', 'right')),
                    os.path.join(args.output_dir, organ.replace('_left', 's'))
                )
    finally:
        try:
            shutil.rmtree(temp_ct_good)
            shutil.rmtree(temp_ct_bad)
            shutil.rmtree(temp_mask_good)
            shutil.rmtree(temp_mask_bad)
            #shutil.rmtree(os.path.join(args.output_dir, 'good_projection'))
            #shutil.rmtree(os.path.join(args.output_dir, 'bad_projection'))
        except:
            pass

if __name__ == '__main__':
    main()


#python3 ProjectDatasetFlex_single.py \
#  --ct_good path/to/ct1.nii.gz \
#  --mask_good path/to/masks1/ \
#  --ct_bad path/to/ct2.nii.gz \
#  --mask_bad path/to/masks2/ \
#  --output_dir ./comparison_results
'''
python3 ProjectDatasetFlex_single.py \
  --ct_good /path/to/ct_data/case_01/ct.nii.gz \
  --mask_good /path/to/ct_data/case_01/outputs/segmentations \
  --ct_bad /path/to/ct_data/case_01/ct.nii.gz \
  --mask_bad /path/to/ct_data/case_01/outputs/segmentations \
  --output_dir ./comparison_results
'''