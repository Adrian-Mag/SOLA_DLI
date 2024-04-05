import os
import pandas as pd


# Function to generate full filenames
def generate_filenames(mode, kernel_type, data_directory):
    filename = f"{kernel_type}-sens_{mode}_iso.dat"
    return filename


def load_normal_data(kernel_type, data_directory):
    modes_df = pd.read_csv(os.path.join(data_directory,'data_list_SP12RTS'),
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
                         sep='\s+',
                         header=None,
                         names=['Radius', 'Sensitivity'])
        if domain is None:
            domain = df['Radius'].values
        kernels.append(df['Sensitivity'].values)
    return domain, kernels
