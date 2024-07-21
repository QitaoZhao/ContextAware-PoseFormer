cd dataset/mpi_inf_3dhp

# Download raw videos and annotations
bash get_dataset.sh
bash get_testset.sh

mv mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set ../
rm -r mpi_inf_3dhp_test_set

# Prepare labels
cd ../../
if [ ! -f "dataset/data_train_3dhp.npz" ]; then
    python dataset/data_util/data_to_npz_3dhp.py
fi

if [ ! -f "dataset/data_test_3dhp.npz" ]; then
    python dataset/data_util/data_to_npz_3dhp_test.py
fi

# Convert raw videos to images for the training set
python dataset/data_util/video_to_images.py

# Crop images to a smaller size (256x192)
python dataset/data_util/convert_to_small.py
python dataset/data_util/convert_to_small_test.py