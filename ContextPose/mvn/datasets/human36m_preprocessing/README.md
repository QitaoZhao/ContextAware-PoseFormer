Human3.6M preprocessing scripts
=======

These scripts help preprocess Human3.6M dataset so that it can be used with `class Human36MMultiViewDataset`.

Here is how we do it (brace yourselves):

0. Make sure you have a lot (around 200 GiB?) of free disk space. Otherwise, be prepared to always carefully delete intermediate files (e.g. after you extract movies, delete the archives).

1. Please do not ask me for a copy of the Human3.6M dataset. I do not own the data, nor do I have permission to redistribute it. You will need to create an account at
   http://vision.imar.ro/human3.6m/ to download the dataset and put the raw data under `$SOMEWHERE_ELSE/h36m_raw_data`
   
2. Choose a folder for the processed data to be stored. Make it accessible like `$TransPose_ROOT/data/human36m` where `ROOT_PATH` is `data/human36m` by default. Modify `RAW_DATA_PATH` to be `$SOMEWHERE_ELSE/h36m_raw_data`, and `NUM_PROCESS` to be the one your environment allows in the first two lines in `process_h36m.sh`.

3. Download `h36m.zip` (provided by [Julieta Martinez](https://github.com/una-dinosauria/3d-pose-baseline/)) from [Google Drive](https://drive.google.com/file/d/1PIqzOfdIYUVJudV5f22JpbAqDCTNMU8E/view?usp=sharing)
    ```bash
    mkdir -p $TransPose_ROOT/data/human36m/extra/una-dinosauria-data
    ```
    and put the zipfile under `$TransPose_ROOT/data/human36m/extra/una-dinosauria-data`.

4. We have integrated all the processing codes into `process_h36m.sh` and you can simply run it under `$TransPose_ROOT` to finish all the procedures instructed by [mvn/datasets/human36m_preprocessing/README.md](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md):

    ```bash
    cd $TransPose_ROOT
    # in our environment, this took around 5 hours with 40 processes
    sh process_h36m.sh
    ```

5. If everything goes well, your `data` directory tree will be like this (our code also support read .zip data, you can simply zip the `processed` folder to be `processed.zip`, and change the **data_format** in the config file to be 'zip'.):
    ```
    $TransPose_ROOT
    |-- data
    `-- |-- human36m
        `-- |-- extra
            |   |-- una-dinosauria-data
            |   `-- |-- h36m.zip 
            |   |   |-- h36m   
            |   |   `-- |-- S1
            |   |       |-- S11
            |   |       |-- S5
            |   |       |-- S6
            |   |       |-- S7
            |   |       |-- S8
            |   |       |-- S9
            |   |       `-- cameras.h5
            |   |-- bboxes-Human36M-GT.npy
            |   |-- human36m-multiview-labels-GTbboxes.npy
            |   `-- mean_and_std_limb_length.h5
            |-- extracted
            |   |-- S1
            |   `-- |-- Poses_D2_Positions
            |   |   |-- Poses_D3_Positions
            |   |   |-- Poses_D3_Positions_mono
            |   |   |-- Poses_D3_Positions_mono_universal
            |   |   `-- Videos
            |   |-- S11
            |   |-- S5
            |   |-- S6
            |   |-- S7
            |   |-- S8
            |   `-- S9
            `-- processed (or processed.zip)
                |-- S1
                `-- |-- Directions-1
                |   `-- |-- imageSequence
                |   |   |-- imageSequence-undistorted
                |   |   `-- |-- 54138969
                |   |   |   `-- |-- img_000001.jpg      
                |   |   |   |-- |-- img_000008.jpg      
                |   |   |   |-- |-- img_000011.jpg      
                |   |   |   `-- |-- ...
                |   |   |   |-- 55011271
                |   |   |   |-- 58860488
                |   |   |   `-- 60457274
                |   |   `-- annot.h5
                |   |-- Directions-2
                |   |-- Discussion-1
                |   `-- ...
                |-- S11
                |-- S5
                |-- S6
                |-- S7
                |-- S8
                `-- S9    
    ```

6. [If you want to learn more for processing the data] All the detailed instructions in `process_h36m.sh` can be found at [mvn/datasets/human36m_preprocessing/README.md](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md).