import os
import sys
import zipfile
import shutil

# put your folder path where all zip files are
root_path = '/media/argha/data/LINUX/raptorX_all'
# no need to edit
work_path = '/home/argha/WORK/extracted_data/extracted_data/2D_data/INTRA_cntct_map'


def process():
    files = os.listdir(root_path)
    # filter out only zip files
    files = [file for file in files if file.endswith('.zip')]

    # make a directory where to put all the work
    try:
        os.mkdir(work_path)
    except:
        pass

    # process each zip one by one
    for file in files:
        file_path = os.path.join(root_path, file)

        # unzip file in memory and excract the target file
        with zipfile.ZipFile(file_path) as zip_file:
            for member in zip_file.namelist():
                filename = os.path.basename(member)

                # filter out desired file
                if 'contactmap.txt' not in filename:
                    continue

                # copy file (taken from zipfile's extract)
                source = zip_file.open(member)
                # removing .zip and add .txt to create new file name from zip file name
                new_file_name = f'{file.split(".")[0]}.txt'
                target = open(os.path.join(work_path, new_file_name), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

                # work done
                break




process()