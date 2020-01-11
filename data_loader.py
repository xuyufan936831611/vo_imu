from __future__ import division
import os
import random
import tensorflow as tf

class DataLoader(object):
    def __init__(self, 
                 dataset_dir_l=None,
                 dataset_dir_r=None,
                 test_dataset_dir=None,
                 batch_size=None, 
                 img_height=None, 
                 img_width=None, 
                 num_source=None,
                 num_scales=None
                 ):
        self.dataset_dir_l = dataset_dir_l
        self.dataset_dir_r = dataset_dir_r
        self.test_dataset_dir = test_dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = 4

    def load_train_batch(self):
        """Load a batch of training instances.
        """

        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list_l = self.format_file_list(self.dataset_dir_l, 'train')
        image_paths_queue_l = tf.train.string_input_producer(
            file_list_l['image_file_list'],
            seed=seed, 
            shuffle=True)
        file_list_r = self.format_file_list(self.dataset_dir_r, 'train')
        image_paths_queue_r = tf.train.string_input_producer(
            file_list_r['image_file_list'],
            seed=seed,
            shuffle=True)

        cam_paths_queue_l = tf.train.string_input_producer(
            file_list_l['cam_file_list'],
            seed=seed, 
            shuffle=True)
        cam_paths_queue_r = tf.train.string_input_producer(
            file_list_r['cam_file_list'],
            seed=seed,
            shuffle=True)
        imu_paths_queue_l = tf.train.string_input_producer(
            file_list_l['imu_file_list'],
            seed=seed,
            shuffle=True)
        file_list_mask = self.format_file_list_mask(self.dataset_dir_l, 'train')
        mask_paths_queue_l = tf.train.string_input_producer(
            file_list_mask['mask_file_list'],
            seed=seed,
            shuffle=True)
        self.steps_per_epoch = int(
            len(file_list_l['image_file_list'])//self.batch_size)
        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents_l = img_reader.read(image_paths_queue_l)  # (It-1,It,It+1)
        _, image_contents_r = img_reader.read(image_paths_queue_r)  # (It)
        _, mask_contents_l = img_reader.read(mask_paths_queue_l)

        image_seq_l = tf.image.decode_jpeg(image_contents_l)
        image_seq_r = tf.image.decode_jpeg(image_contents_r)
        mask_seq_l = tf.image.decode_jpeg(mask_contents_l)


        Its1_l,It_l, Itp1_l = self.unpack_image_sequence_l(image_seq_l, self.img_height, self.img_width,self.num_source)
        It_r = self.unpack_image_sequence_r(image_seq_r, self.img_height, self.img_width)
        mask_t_l = self.unpack_mask_sequence_l(mask_seq_l, self.img_height, self.img_width)
        mask_t_l = tf.reshape(mask_t_l, [self.img_height, self.img_width, 1])
        # Load camera intrinsics
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents_l = cam_reader.read(cam_paths_queue_l)#(3*4),(BF,0,0)'
        _, raw_imu_contents_l1 = cam_reader.read(imu_paths_queue_l)
        rec_def_l = []
        for i in range(10):
            rec_def_l.append([1.])   #[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
        raw_cam_vec_l = tf.decode_csv(raw_cam_contents_l,
                                    record_defaults=rec_def_l)
        raw_cam_vec_l = tf.stack(raw_cam_vec_l[0:10])

        BF_l = raw_cam_vec_l[9]
        BF_l = tf.reshape(BF_l,[1])
        raw_cam_vec_l = tf.stack(raw_cam_vec_l[0:9])
        intrinsics = tf.reshape(raw_cam_vec_l, [3, 3])

        _, raw_cam_contents_r = cam_reader.read(cam_paths_queue_r)
        rec_def_r = []
        for i in range(10):
            rec_def_r.append([1.])  # [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
        raw_cam_vec_r = tf.decode_csv(raw_cam_contents_r,
                                      record_defaults=rec_def_r)
        raw_cam_vec_r = tf.stack(raw_cam_vec_r[0:10])
        BF_r = raw_cam_vec_r[9]
        BF_r = tf.reshape(BF_r, [1])

        BF=tf.abs(BF_r)+tf.abs(BF_l)       #only one scale
        rec_def_l = []
        for i in range(6):
            rec_def_l.append([1.])

        raw_imu_vec_l1 = tf.decode_csv(raw_imu_contents_l1,
                                       record_defaults=rec_def_l)
        imu_vec = tf.stack([raw_imu_vec_l1[0:6]])

        # Data augmentation
        Its1_l, It_l,It_r ,Itp1_l  = self.data_augmentation(Its1_l, It_l,It_r , Itp1_l)
        # Form training batches
        Its1_l,It_l, It_r,Itp1_l ,BF,intrinsics,mask_t_l,imu_vec = \
            tf.train.batch([Its1_l,It_l,It_r,Itp1_l , BF, intrinsics,mask_t_l,imu_vec],
                           batch_size=self.batch_size)

        intrinsics = self.get_multi_scale_intrinsics(intrinsics, self.num_scales)
        tgt_image = It_l
        if self.num_source==4:
            src_image_stack = tf.concat([Its1_l[:,:,:self.img_width,:], Its1_l[:,:,self.img_width:,:],
                                           Itp1_l[:,:,:self.img_width,:], Itp1_l[:,:,self.img_width:,:]] ,3)
        if self.num_source == 2:
            src_image_stack = tf.concat( [Its1_l, Itp1_l],3)
        return  tgt_image,src_image_stack,It_r,BF,intrinsics,mask_t_l,imu_vec

    def load_test_batch(self):
        """Load a batch of training instances.
        """

        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list_l = self.format_file_list(self.test_dataset_dir, 'test')
        image_paths_queue_l = tf.train.string_input_producer(
            file_list_l['image_file_list'],
            seed=seed,
            shuffle=False)

        cam_paths_queue_l = tf.train.string_input_producer(
            file_list_l['cam_file_list'],
            seed=seed,
            shuffle=True)

        self.steps_per_epoch = int(
            len(file_list_l['image_file_list'])//self.batch_size)
        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents_l = img_reader.read(image_paths_queue_l)  # (It-1,It,It+1)
        image_seq_l = tf.image.decode_jpeg(image_contents_l)

        It_l= self.unpack_image_sequence_r(image_seq_l, self.img_height, self.img_width)

        # Form training batches
        It_l = tf.train.batch([It_l],batch_size=self.batch_size)
        tgt_image = It_l
        return  tgt_image
    def make_intrinsics_matrix(self, fx, fy, cx, cy):

        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, Its1_l, It_l, It_r, Itp1_l):

        def augment_image_pair(Its1_l,It_l, It_r, Itp1_l):

            random_gamma = tf.random_uniform([], 0.8, 1.2)
            Its1_l=tf.cast(Its1_l,tf.float32)
            It_l=tf.cast(It_l,tf.float32)
            It_r = tf.cast(It_r, tf.float32)
            Itp1_l = tf.cast(Itp1_l, tf.float32)
            left_image_aug_t = pow(Its1_l, random_gamma)
            right_image_aug_t = pow(It_l, random_gamma)
            left_image_aug_tp1 = pow(It_r, random_gamma)
            right_image_aug_tp1 = pow(Itp1_l, random_gamma)
            # randomly shift brightness
            random_brightness = tf.random_uniform([], 0.5, 2.0)
            left_image_aug_t = left_image_aug_t * random_brightness
            right_image_aug_t = right_image_aug_t * random_brightness
            left_image_aug_tp1 = left_image_aug_tp1 * random_brightness
            right_image_aug_tp1 = right_image_aug_tp1 * random_brightness

            # randomly shift color
            random_colors = tf.random_uniform([3], 0.8, 1.2)
            white = tf.ones([tf.shape(Its1_l)[0], tf.shape(Its1_l)[1]])
            color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
            left_image_aug_t *= color_image
            right_image_aug_t *= color_image
            left_image_aug_tp1 *= color_image
            right_image_aug_tp1 *= color_image

            # saturate
            left_image_aug_t = tf.clip_by_value(left_image_aug_t, 0, 255)
            right_image_aug_t = tf.clip_by_value(right_image_aug_t, 0, 255)
            left_image_aug_tp1 = tf.clip_by_value(left_image_aug_tp1, 0, 255)
            right_image_aug_tp1 = tf.clip_by_value(right_image_aug_tp1, 0, 255)

            left_image_aug_t = tf.floor(left_image_aug_t)
            right_image_aug_t = tf.floor(right_image_aug_t)
            left_image_aug_t = tf.cast(left_image_aug_t, tf.uint8)
            right_image_aug_t = tf.cast(right_image_aug_t, tf.uint8)

            left_image_aug_tp1 = tf.floor(left_image_aug_tp1)
            right_image_aug_tp1 = tf.floor(right_image_aug_tp1)
            left_image_aug_tp1 = tf.cast(left_image_aug_tp1, tf.uint8)
            right_image_aug_tp1 = tf.cast(right_image_aug_tp1, tf.uint8)
            return left_image_aug_t, right_image_aug_t,left_image_aug_tp1, right_image_aug_tp1

        # batch_size = Its1_l.get_shape().as_list()[0]
        height=Its1_l.get_shape().as_list()[0]
        width=Its1_l.get_shape().as_list()[1]
        channel=Its1_l.get_shape().as_list()[2]
        do_augment = tf.random_uniform([], 0, 1)
        Its1_l, It_l,It_r, Itp1_l = tf.cond(do_augment <0.5,       \
                                            lambda:augment_image_pair(Its1_l, It_l, It_r, Itp1_l),
                                            lambda: (Its1_l, It_l, It_r, Itp1_l))
        Its1_l = tf.reshape(Its1_l,[height,width,channel])
        It_l = tf.reshape(It_l, [height, width, channel])
        It_r = tf.reshape(It_r, [height, width, channel])
        Itp1_l = tf.reshape(Itp1_l, [height, width, channel])

        return Its1_l, It_l,It_r, Itp1_l


    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        imu_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_imu.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        all_list['imu_file_list'] = imu_file_list
        return all_list
    def format_file_list_mask(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        mask_file_list = [os.path.join(data_root, 'mask_tgt',subfolders[i],
            frame_ids[i] + '.png') for i in range(len(frames))]

        all_list = {}
        all_list['mask_file_list'] = mask_file_list

        return all_list
    def unpack_image_sequence_l(self, image_seq, img_height, img_width, num_source):
        # Assuming the first image is the target frame
        tgt_start_idx = int(img_width * (num_source // 2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source // 2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source // 2)), -1])
        tgt_image.set_shape([img_height, img_width, 3])
        src_image_1.set_shape([img_height, img_width*(self.num_source//2), 3])
        src_image_2.set_shape([img_height, img_width*(self.num_source//2), 3])
        return src_image_1,tgt_image, src_image_2

    def unpack_image_sequence_r(self, image_seq, img_height, img_width):
        # Assuming the first image is the target frame
        #It_start_idx = 0
        #It
        It_image = image_seq
        It_image.set_shape([img_height, img_width, 3])
        return It_image
    def unpack_mask_sequence_l(self, image_seq, img_height, img_width):
        # Assuming the first image is the target frame
        #It_start_idx = 0
        #It
        It_image = image_seq
        It_image.set_shape([img_height, img_width, 1])
        return It_image
    #for test
    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, 0, tgt_start_idx, 0], 
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0, 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, 0, int(tgt_start_idx + img_width), 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2],2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, 0, i*img_width, 0], 
                                    [-1, -1, img_width, -1]) 
                                    for i in range(num_source)],3)
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
