PROJECT_PATH	:= $(shell pwd)
MSRA_SRC	:= https://www.dropbox.com/s/bm2xg4uxst44f9d/msra15_o_l_r_d_u_ld_rd_lu_ru_L_30_post_31.h5?dl=0#
MSRA_DST	:= $(PROJECT_PATH)/msra15_o_l_r_d_u_ld_rd_lu_ru_L_30_post_31.h5
MNIST_SRC	:= https://github.com/BarclayII/tracking-with-rnn/raw/master/mnist.h5
MNIST_DST	:= $(PROJECT_PATH)/mnist.h5

dependencies:
	pip install pillow tqdm h5py scipy numpy glog opencv-python

download:
	wget -O $(MSRA_DST) $(MSRA_SRC)
	wget -O $(MNIST_DST) $(MNIST_SRC)

msra: $(MSRA_DST)
	python3 main_8view_label.py \
	--log_root $(PROJECT_PATH) \
	--model_root $(PROJECT_PATH) \
	--h5_path $(MSRA_DST) \
	--model gan

mnist: $(MNIST_DST)
	python3 main_mnist.py \
	--log_root $(PROJECT_PATH) \
	--model_root $(PROJECT_PATH) \
	--depth_height 32 \
	--depth_width 32 \
	--batch_size 16 \
	--h5_path $(MNIST_DST)
