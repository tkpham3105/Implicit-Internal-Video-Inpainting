import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

import data.dataloader as dl 
from config.config import Config
from model.inpaint_model import RefineModel, BaseModel, StackModel
from loss import ambiguity_loss, stable_loss, disparity_loss

from DSMNet.models.DSMNet2x2 import DSMNet
from torch.autograd import Variable
import torch
import torch.nn as nn
import skimage
import cv2
import shutil
import numpy as np

#if __name__ == "__main__":
    
def main(FLAGS):

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID 
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    
    # define the model
    if FLAGS.coarse_only:
        model = BaseModel()
    else:
        if FLAGS.use_refine:
            model = RefineModel()
        else: 
            model = StackModel()
    
    continue_step = -1
    if not FLAGS.model_restore=="":
        model.load_weights(FLAGS.model_restore)
        continue_step = int(FLAGS.model_restore.split("_")[-1])

    if FLAGS.ambiguity_loss:
        vgg = keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer('block4_conv2').output]
        amb_model = keras.Model([vgg.input], outputs)
        amb_model.trainable = False

    if FLAGS.disparity_loss:
        disparity_model = DSMNet(FLAGS.max_disp)
        disparity_model = nn.DataParallel(disparity_model, device_ids = [1]).cuda(FLAGS.gpu_disp)
        checkpoint = torch.load(FLAGS.checkpoint_disp, map_location=lambda storage, loc: storage.cuda(FLAGS.gpu_disp))
        disparity_model.load_state_dict(checkpoint['state_dict'], strict=False)        
        
    # define the optimizer
    optimizer = keras.optimizers.Adam(learning_rate=FLAGS.lr, beta_1=0.9, beta_2=0.999)

    # define the dataloader
    with tf.device("gpu:6"):
        dir_mask = f"{FLAGS.dir_mask}/../masks_dilated" if FLAGS.dilate_mask else FLAGS.dir_mask 
        full_ds = dl.build_dataset_video_stereo(FLAGS.dir_video, dir_mask, dir_mask, 
                                FLAGS.batch_size, FLAGS.max_epochs, FLAGS.img_shapes[0], FLAGS.img_shapes[1])
    #dist_full_ds = mirrored_strategy.experimental_distribute_dataset(full_ds)
    
    #summary writer
    writer = tf.summary.create_file_writer(FLAGS.log_dir)

    # define the training steps and loss
    def training_step(batch_data, step):       
        batch_pos = batch_data[0]
        mask1 = batch_data[2] 
        mask2 = batch_data[1] 
        shift_h = tf.random.uniform(shape=[], maxval=mask1.shape[1], dtype=tf.int64)
        shift_w = tf.random.uniform(shape=[], maxval=mask1.shape[2], dtype=tf.int64)
        mask1 = tf.roll(mask1, (shift_h, shift_w), axis=(1,2))  
        mask = tf.cast(
            tf.logical_or(
                tf.cast(mask1, tf.bool),
                tf.cast(mask2, tf.bool),
            ),
            tf.float32
        )
        batch_incomplete = batch_pos*(1.-mask)
        xin = batch_incomplete
        x = tf.concat([xin, mask], axis=3)

        # stabilization loss
        if FLAGS.stabilization_loss:
            with tf.device("gpu:6"):
                T = stable_loss.get_transform(FLAGS)

                # Perform transformation
                T_batch_pos = tfa.image.transform(batch_pos, T, interpolation = 'BILINEAR')
                Tmask = tfa.image.transform(mask, T, interpolation = 'NEAREST')
                Tmask2 = tfa.image.transform(mask2, T, interpolation = 'NEAREST')
                Tmask_n = tf.cast(
                    tf.logical_or(
                        tf.cast(mask2, tf.bool),
                        tf.cast(Tmask2, tf.bool),),
                    tf.float32)
            
                Tx = tf.concat([T_batch_pos*(1-Tmask), Tmask], axis=3)
 
        stab_loss, amb_loss, disp_loss = 0, 0, 0
        with tf.GradientTape(persistent=True) as tape:
            if not FLAGS.coarse_only:
                x1, x2 = model(x, mask) 
                base_loss = FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x1)*(1-mask2))
                base_loss += FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x2)*(1-mask2))
                loss = base_loss
                
                if FLAGS.stabilization_loss:
                    with tf.device("gpu:6"):
                        Tx1, Tx2 = model(Tx, Tmask)
                        stab_loss = FLAGS.stabilization_loss_alpha * tf.reduce_mean(tf.abs((Tx2 - x2)-(T_batch_pos-batch_pos)) * (1-Tmask_n))
                        stab_loss += FLAGS.stabilization_loss_alpha * tf.reduce_mean(tf.abs((Tx1 - x1)-(T_batch_pos-batch_pos)) * (1-Tmask_n))
                        loss += stab_loss
                        del Tx1, Tx2, Tx, Tmask 
              
                if FLAGS.ambiguity_loss:
                    with tf.device("gpu:2"):
                    #loss += FLAGS.ambiguity_loss_alpha*ambiguity_loss.perceptual_loss((1-mask2)*x2, (1-mask2)*batch_pos)
                        amb_loss =  FLAGS.ambiguity_loss_alpha*ambiguity_loss.contextual_loss((1-mask2[::-1,:,:,:])*x2, (1-mask2[::-1,:,:,:])*batch_pos[::-1,:,:,:], amb_model)
                        loss += amb_loss
                
                if FLAGS.disparity_loss:
                    left_inpaint = (x2 + 1) * 127.5
                    left = (batch_data[0] + 1) * 127.5
                    right = (batch_data[3] + 1) * 127.5

                    left, h, w = disparity_loss.preprocess(left, FLAGS.crop_height, FLAGS.crop_width)
                    right, _, _ = disparity_loss.preprocess(right, FLAGS.crop_height, FLAGS.crop_width)
                    left_inpaint, _, _ = disparity_loss.preprocess(left_inpaint, FLAGS.crop_height, FLAGS.crop_width)
                    
                    left = left.cuda(FLAGS.gpu_disp)
                    right = right.cuda(FLAGS.gpu_disp)
                    left_inpaint = left_inpaint.cuda(FLAGS.gpu_disp)
                    
                    disparity_model.eval()
                    with torch.no_grad():
                        disparity_GT = disparity_model(left, right).detach().cpu().unsqueeze(3)
                        disparity_IP = disparity_model(left_inpaint, right).detach().cpu().unsqueeze(3)
                        if h <= FLAGS.crop_height and w <= FLAGS.crop_width:
                            disparity_GT = disparity_GT[:, FLAGS.crop_height - h: FLAGS.crop_height, FLAGS.crop_width - w: FLAGS.crop_width]
                            disparity_IP = disparity_IP[:, FLAGS.crop_height - h: FLAGS.crop_height, FLAGS.crop_width - w: FLAGS.crop_width]
                    disp_loss = FLAGS.disparity_loss_alpha * tf.reduce_mean(tf.abs(disparity_GT - disparity_IP) * (1-mask2))
                    loss += disp_loss
                    del disparity_GT, disparity_IP
                        #skimage.io.imsave("tmp/GT.png", (disparity_GT[0].detach().cpu().numpy() * 256).astype('uint16'))
                        #skimage.io.imsave("tmp/IP.png", (disparity_IP[0].detach().cpu().numpy() * 256).astype('uint16'))                       
                          
            else:
                x1 = model(x) 
                loss = FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x1)*(1-mask2))
                x2 = x1

        #if FLAGS.disparity_loss:
        #    left = (batch_data[0] + 1) * 127.5
        #    right = (batch_data[3] + 1) * 127.5
                
        #    left, right, h, w = disparity_loss.preprocess(left, right, FLAGS.crop_height, FLAGS.crop_width)
        #    left = left.cuda(FLAGS.gpu_disp)
        #    right = right.cuda(FLAGS.gpu_disp)

        #    disparity_model.eval()
        #    with torch.no_grad():
        #        disparity_GT = disparity_model(left, right) 
        #        if h <= FLAGS.crop_height and w <= FLAGS.crop_width:
        #            disparity_GT = disparity_GT[:, FLAGS.crop_height - h: FLAGS.crop_height, FLAGS.crop_width - w: FLAGS.crop_width]                  
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


        # add summary
        batch_complete = x2*mask + batch_incomplete*(1.-mask)
        viz_img = [batch_pos, batch_incomplete, x1, x2, batch_complete]
        viz_img_concat = (tf.concat(viz_img, axis=2) + 1) / 2.0  

        # a work around here / since there is a bug in tf image summary until tf 2.3
        if step % FLAGS.summary_iters == 0:
            with tf.device("cpu:0"):
                with writer.as_default():
                    tf.summary.image('input_input_x1_x2_output', viz_img_concat, step=step, max_outputs=6)
                    tf.summary.scalar('loss', loss, step=step)
                    #tf.summary.scalar('base_loss', base_loss, step=step)
                    #tf.summary.scalar('stable_loss', stable_loss, step=step)
                    #tf.summary.scalar('amb_loss', amb_loss, step=step)
                    #tf.summary.scalar('disp_loss', disp_loss, step=step)
        return loss, base_loss, stab_loss, amb_loss, disp_loss
    

#    @tf.function
#    def distributed_train_step(dataset_inputs, step):
#        per_replica_losses = mirrored_strategy.experimental_run_v2(training_step, args=(dataset_inputs, step,))
#        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # start training
    for step, batch_data in enumerate(full_ds):
        if step <= continue_step: 
            print(f"Skipped step {step}")
            continue
        #torch.cuda.empty_cache()
        step = tf.convert_to_tensor(step, dtype=tf.int64)
        losses, base_losses, stable_losses, amb_losses, disp_losses = training_step(batch_data, step)

        if step % FLAGS.print_iters == 0:
            print("Step:", tf.get_static_value(step), \
		  "Sum", tf.get_static_value(losses), \
                  "Base", tf.get_static_value(base_losses), \
                  "Stable", tf.get_static_value(stable_losses), \
                  "Amb", tf.get_static_value(amb_losses), \
                  "Disp", tf.get_static_value(disp_losses))

        if step % FLAGS.summary_iters == 0:
            writer.flush()

        if step % FLAGS.model_iters == 0:
            model.save_weights("%s/checkpoint_%d"%(FLAGS.log_dir, step.numpy()))

        if step >= FLAGS.max_iters:
            break

        del losses, base_losses, stable_losses, amb_losses, disp_losses
        #print(f"Step {step}")
    print('finished!')

if __name__=="__main__":
    FLAGS = Config('config/train_stereo.yml')
    if FLAGS.dilate_mask: 
        mask_folder = FLAGS.dir_mask
        mask_dilated_folder = f"{mask_folder}/../masks_dilated"
        os.makedirs(mask_dilated_folder, exist_ok=True)
        for side in ["L", "R"]:
            os.makedirs(f"{mask_dilated_folder}/{side}", exist_ok=True)
            for i, f in enumerate(os.listdir(f"{mask_folder}/{side}")):
                mask_path = f"{mask_folder}/{side}/{f}"
                mask = cv2.imread(mask_path, 0)
                kernel = np.ones((9,9),np.uint8)
                mask = cv2.dilate(mask,kernel,iterations = 1)
                mask[mask >0]=1.
                mask = mask*255
                mask_dilated_path = f"{mask_dilated_folder}/{side}/{f}"
                cv2.imwrite(mask_dilated_path, mask)
    main(FLAGS)
    if FLAGS.dilate_mask:
        shutil.rmtree(mask_dilated_folder)
    
    #p = multiprocessing.Process(target=main)
    #p.start()
    #p.join()
    #disparity_model = DSMNet(192)
    #disparity_model = nn.DataParallel(disparity_model)
