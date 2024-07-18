import os
import subprocess

urlbase = 'https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/'
# List of parcellations to download
parcellations = [
    'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
    'Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
    'Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
    'Schaefer2018_300Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
    'Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
    'Schaefer2018_500Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
    'Schaefer2018_600Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
    'Schaefer2018_700Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
    'Schaefer2018_800Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
    'Schaefer2018_900Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
]

# Directory to save the downloaded files
output_dir = './Centroid_coordinates/'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to download a file
def download_file(parcel, urlbase, output_dir):
    filename = os.path.join(output_dir, parcel)
    print('filename',filename)
    url = os.path.join(urlbase, parcel)
    print('url',url)
    command = [
        'wget',
        '--no-check-certificate',
        '--content-disposition',
        url,
        '-P', output_dir
    ]
    subprocess.run(command)
    print(f"Downloaded {filename}")

# Download each file in the list
for parcel in parcellations:
    download_file(parcel, urlbase, output_dir)

print("All files downloaded.")
