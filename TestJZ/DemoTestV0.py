#checking packages
import os
import sys

import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# appending path for ukbb_cardiac
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

def remove_slices_time (image_path, output_path, step=4):
    # Load the NIfTI image
    img = nib.load(image_path)
    img_data = img.get_fdata()

    # Get the shape of the image data
    original_shape = img_data.shape

    # Remove slices
    modified_data_tmp = np.delete(img_data, np.arange(step-1, img_data.shape[2], step), axis=2)
    
    # Remove time
    modified_data = np.delete(modified_data_tmp, np.arange(step-1, img_data.shape[3], step), axis=3)

    # Create a new NIfTI image with the modified data
    modified_img = nib.Nifti1Image(modified_data, img.affine, img.header)

    # Save the modified image
    nib.save(modified_img, output_path)

    print(f"Original shape: {original_shape}")
    print(f"Modified shape: {modified_data.shape}")
    print(f"Modified image saved to {output_path}")

def view_slice(nifti_file_path, slice=1, time=2, d3=False, output_image_path='first_slice.png'):
    # Load the NIfTI file
    nifti_img = nib.load(nifti_file_path)
    
    # Get the image data as a NumPy array
    nifti_data = nifti_img.get_fdata()
    
    # Extract the first slice (assuming the slices are in the third dimension)
    if not d3: first_slice = nifti_data[:, :, slice, time]
    else: first_slice = nifti_data[:, :, slice]
    
    # Plot the first slice
    plt.imshow(first_slice, cmap='gray')
    plt.title('First Slice of NIfTI Image')
    plt.axis('off')  # Hide axis labels
    
    # Save the plot to a file
    plt.savefig(output_image_path)
    print(f"First slice image saved to {output_image_path}")

# # tensor 2.x
# def list_available_gpus():
#     # Get the list of available GPU devices
#     gpus = tf.config.list_physical_devices('GPU')
#     gpu_indices = []
#     if gpus:
#         print("GPUs available:")
#         for i, gpu in enumerate(gpus):
#             print(f"  - GPU {i}: {gpu.name}")
#             gpu_indices.append(i)
#     else:
#         print("No GPUs found.")
    
#     return gpu_indices

# for tensor 1.x
def list_available_gpus():
    from tensorflow.python.client import device_lib

    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
    gpu_indices = get_available_gpus()
    
    if gpu_indices:
        print("GPUs available:")
        for i, gpu in enumerate(gpu_indices):
            print(f"  - GPU {i}: {gpu}")
    else:
        print("No GPUs found.")
    
    return gpu_indices

# find file position in docker image
def find_file(filename, search_path):
    result = []
    # Iterate over the directory tree
    for root, dirs, files in os.walk(search_path):
        # Check if the file is in the current directory
        if filename in files:
            # Append the full path of the file to the result list
            result.append(os.path.abspath(os.path.join(root, filename)))
    return result

def SA_Segmentation(args_sa):
    print('Deploying the segmentation network ...')
    
    command = f"CUDA_VISIBLE_DEVICES={args_sa.cuda_device} " +\
                f"{args_sa.run_machine} " +\
                f"{args_sa.network_file} " +\
                f"--seq_name {args_sa.seq_name} " +\
                f"--data_dir {args_sa.data_dir} " +\
                f"--model_path {args_sa.model_path}"

    os.system(command)
    
def LA_Segmentation(args_la):
    
    command = f"CUDA_VISIBLE_DEVICES={args_la.cuda_device} " +\
                f"{args_la.run_machine} " +\
                f"{args_la.network_file} " +\
                f"--seq_name {args_la.seq_name} " +\
                f"--data_dir {args_la.data_dir} " +\
                f"--model_path {args_la.model_path}"
    
    print('Deploying the segmentation network ...')
    os.system(command)


if __name__ == '__main__':
    # Single gpu setting
    gpu_indice = list_available_gpus()[0]
    
    # Get the current working directory
    current_working_directory = os.getcwd()
    print(f"Current working directory: {current_working_directory}")
    
    # find files
    filename = "FCN_sa.meta"

    # Search for the exact file name
    file_paths = find_file(filename, current_working_directory)
    for file_path in file_paths:
        print(f"Found file at: {file_path}")
        
    # # shrink image size wont work for net work!! 
    # input_image_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/sa_ori.nii.gz'
    # output_image_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/sa.nii.gz'
    # remove_slices_time(input_image_path, output_image_path, step=2)
    
    # check results
    nifti_file_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/sa.nii.gz'
    out_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/sa_slice8_time6.png'
    view_slice(nifti_file_path, slice=8, time=6, output_image_path=out_path)
    
    nifti_file_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/seg_sa.nii.gz'
    out_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/seg_sa_slice8_time6.png'
    view_slice(nifti_file_path, slice=8, time=6, output_image_path=out_path)
    
    # check results
    nifti_file_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/sa_ED.nii.gz'
    out_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/sa_ED_time8.png'
    view_slice(nifti_file_path, slice=8, d3=True, output_image_path=out_path)
    
    nifti_file_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/seg_sa_ED.nii.gz'
    out_path = '/workspaces/ukbb_cardiac/TestJZ/demo_image/1/seg_sa_ED_time8.png'
    view_slice(nifti_file_path, slice=8, d3=True, output_image_path=out_path)
    
    # sys.exit()
    
    # # Analyse show-axis images
    # print('******************************')
    # print('  Short-axis image analysis')
    # print('******************************')
    
    # # setting up arguments
    # parser_sa = argparse.ArgumentParser(description='Short-axis image analysis')
    # parser_sa.add_argument('--seq_name', type=str, default='sa', help='Sequence name')
    # parser_sa.add_argument('--data_dir', type=str, default='/workspaces/ukbb_cardiac/TestJZ/demo_image', help='Data directory')
    # parser_sa.add_argument('--model_path', type=str, default='/workspaces/ukbb_cardiac/TestJZ/trained_model/FCN_sa', help='Path to the model')
    # parser_sa.add_argument('--cuda_device', type=str, default='0', help='CUDA device to use')
    # parser_sa.add_argument('--run_machine', type=str, default='python3', help='CUDA device to use')
    # parser_sa.add_argument('--network_file', type=str, default='common/deploy_network.py', help='CUDA device to use')
    # args_sa = parser_sa.parse_args()
    
    # print(args_sa)
    
    # # Deploy the segmentation network
    # # too large for 3060TI!!!
    # SA_Segmentation(args_sa)
    
    
    # Analyse long-axis images
    print('******************************')
    print('  Long-axis image analysis')
    print('******************************')
    
    # setting up arguments
    parser_la = argparse.ArgumentParser(description='LA image analysis')
    parser_la.add_argument('--seq_name', type=str, default='la_2ch', help='Sequence name')
    parser_la.add_argument('--data_dir', type=str, default='/workspaces/ukbb_cardiac/TestJZ/demo_image', help='Data directory')
    parser_la.add_argument('--model_path', type=str, default='/workspaces/ukbb_cardiac/TestJZ/trained_model/FCN_la_2ch', help='Path to the model')
    parser_la.add_argument('--cuda_device', type=str, default='0', help='CUDA device to use')
    parser_la.add_argument('--run_machine', type=str, default='python3', help='CUDA device to use')
    parser_la.add_argument('--network_file', type=str, default='common/deploy_network.py', help='CUDA device to use')
    args_la = parser_la.parse_args()
    
    # Deploy the segmentation network
    # too large for 3060TI!!!
    LA_Segmentation(args_la)