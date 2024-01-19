import os 
import pandas as pd
import numpy as np 


# Function to generate full filenames
def generate_filenames(mode, kernel_type, data_directory):
    filename = f"{kernel_type}-sens_{mode}_iso.dat"
    return os.path.join(data_directory, filename)

def load_normal_data(kernel_type, data_directory):
    modes_df = pd.read_csv("/home/adrian/PhD/BGSOLA/mysola/normal_data/kernels_modeplotaat_Adrian/data_list_SP12RTS", 
                           header=None, names=["Mode"])
    # Create a new column with the full filenames
    modes_df['filenames'] = modes_df['Mode'].apply(generate_filenames, 
                                                   kernel_type=kernel_type, 
                                                   data_directory=data_directory)
    # Get all the data
    kernels = []
    domain = None
    for file in modes_df['filenames']:
        df = pd.read_csv(os.path.join(data_directory, file), 
                         delim_whitespace=True, 
                         header=None, 
                         names=['Radius', 'Sensitivity'])
        if domain is None:
            domain = df['Radius'].values
        kernels.append(df['Sensitivity'].values)
    return domain, kernels
    