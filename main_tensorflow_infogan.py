import numpy as np
import time
from PIL import Image
import math,os
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from Chinese_inputs import CommonChar, ImageChar


def combine_images(generated_images):
    num = 100 #generated_images.shape[0]
    width = 10 #int(math.sqrt(num))
    height = 10 #int(math.ceil(float(num)/width))
    depth = generated_images.shape[-1]
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1],depth),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images[:num]):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
    return image

class Model():
    def __init__(self,batch_size, sess):
        self.batch_size = batch_size
        self.input_z = tf.placeholder(tf.float32,shape=(batch_size,90))
        self.input_i = tf.placeholder(tf.float32,shape=(batch_size,64,64,1))
        self.input_c = tf.placeholder(tf.float32,shape=(batch_size,10))
        self.input_zc = tf.concat((self.input_z,self.input_c),axis=-1)
        self.g_outputs = self.build_generator(self.input_zc,is_training=True,is_reuse=False)
        self.sample_outputs = self.build_generator(self.input_zc,is_training=False,is_reuse=True)
        d_logits_fake, q_logits_fake = self.build_discriminator(self.g_outputs,is_training=True,is_reuse=False)
        d_logits_real, _ = self.build_discriminator(self.input_i,is_training=True,is_reuse=True)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_logits_real),
                                                                     logits=d_logits_real,name='dreal'))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_logits_fake),
                                                                     logits=d_logits_fake,name='dfake'))

        self.q_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_c,logits=q_logits_fake))

        self.d_loss = (d_loss_fake + d_loss_real) + self.q_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_logits_fake),
                                                                logits=d_logits_fake,name='gfake')) +self.q_loss

        all_vars = tf.trainable_variables()
        self.d_vars = [var for var in all_vars if "discriminator" in var.name]
        self.g_vars = [var for var in all_vars if "generator" in var.name]

        d_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'discriminator')
        with tf.control_dependencies(d_update):
            self.d_optim = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)\
                .minimize(self.d_loss,var_list=self.d_vars)

        g_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'generator')
        with tf.control_dependencies(g_update):
            self.g_optim = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)\
                .minimize(self.g_loss,var_list=self.g_vars)

        sess.run(tf.global_variables_initializer())

    def build_generator(self,input_z,is_training, is_reuse):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)
        def lrelu(x, leak=0.2, name="lrelu"):
            return tf.maximum(x, leak * x,name=name)
        with tf.variable_scope("generator",reuse=is_reuse) as scope:

            with tf.variable_scope("h0"):
                outputs = tf.layers.dense(input_z,512*4*4,kernel_initializer=w_init)
                outputs = tf.reshape(outputs,[-1,4,4,512])

                #outputs = tf.layers.batch_normalization(outputs, training=is_training, gamma_initializer=g_init)
                outputs = tf.nn.tanh(outputs)


            with tf.variable_scope("conv1"):
                outputs = tf.layers.conv2d_transpose(outputs,256,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)

                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
                outputs = lrelu(outputs)


            with tf.variable_scope("conv2"):
                outputs = tf.layers.conv2d_transpose(outputs,128,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)

                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
                outputs = lrelu(outputs)


            with tf.variable_scope("conv3"):
                outputs = tf.layers.conv2d_transpose(outputs,64,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
                outputs = lrelu(outputs)
            '''
            with tf.variable_scope("h0"):
                outputs = tf.layers.dense(input_z,256*8*8,kernel_initializer=w_init)
                outputs = tf.reshape(outputs,[-1,8,8,256])
                outputs = tf.layers.batch_normalization(outputs, training=is_training, gamma_initializer=g_init)
                outputs = tf.nn.relu(outputs)

            with tf.variable_scope("conv1"):
                outputs = tf.layers.conv2d_transpose(outputs,128,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
                #outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
                outputs = tf.nn.relu(outputs)
            with tf.variable_scope("conv2"):
                outputs = tf.layers.conv2d_transpose(outputs,64,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
                #outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
                outputs = tf.nn.relu(outputs)
            '''
            with tf.variable_scope("outputs"):
                outputs = tf.layers.conv2d_transpose(outputs,1,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
                outputs = tf.tanh(outputs)
        return outputs


    def build_discriminator(self,input_i,is_training,is_reuse):
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        def lrelu(x, leak=0.2, name="lrelu"):
            return tf.maximum(x, leak * x,name=name)

        with tf.variable_scope("discriminator",reuse=is_reuse) as scope:
            with tf.variable_scope("conv1"):
                outputs = tf.layers.conv2d(inputs=input_i,filters=64,
                                           kernel_size=(5,5),strides=(2,2),
                                           padding='SAME',activation=None,kernel_initializer=w_init
                                           )
                outputs = tf.nn.tanh(outputs)
            with tf.variable_scope("conv2"):
                outputs = tf.layers.conv2d(inputs=outputs,filters=128,
                                           kernel_size=(5,5),strides=(2,2),
                                           padding='SAME',activation=None,kernel_initializer=w_init
                                           )
                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=gamma_init)
                outputs = lrelu(outputs)


            with tf.variable_scope("conv3"):
                outputs = tf.layers.conv2d(inputs=outputs,filters=256,
                                           kernel_size=(5,5),strides=(2,2),
                                           padding='SAME',activation=None,kernel_initializer=w_init
                                           )
                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=gamma_init)
                outputs = lrelu(outputs)
            '''
            with tf.variable_scope("conv4"):
                outputs = tf.layers.conv2d(inputs=outputs, filters=512,
                                           kernel_size=(5, 5), strides=(2, 2),
                                           padding='SAME', activation=None, kernel_initializer=w_init
                                           )
                #outputs = tf.layers.batch_normalization(outputs, training=is_training, gamma_initializer=gamma_init)
                outputs = lrelu(outputs)
            '''
            outputs = tf.reshape(outputs,[outputs.get_shape()[0].value,-1])
            with tf.variable_scope("out_dis"):
                logits = tf.layers.dense(outputs,1,kernel_initializer=w_init)
            with tf.variable_scope("out_q"):
                outputs = tf.layers.dense(outputs,128,kernel_initializer=w_init)
                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=gamma_init)
                outputs = lrelu(outputs)
                q_logits = tf.layers.dense(outputs,10,kernel_initializer=w_init)
            return logits,q_logits

    def train_one_epoch(self, real_images, z_sample, c_sample, sess, ratio = 1):
        shuffled_images = real_images[np.random.permutation(len(real_images))]
        nb_batch = len(real_images)//self.batch_size
        d_losses = np.zeros(nb_batch)
        g_losses = np.zeros(nb_batch)
        #q_losses = np.zeros(nb_batch)

        start_time = time.time()
        for i in range(nb_batch):

            real_batch = shuffled_images[i*self.batch_size:(i+1)*self.batch_size]
            z =  np.random.normal(loc=0.0,scale=1.0,size=(self.batch_size,90)) #np.random.uniform(-1,1,(model.batch_size,40))
            c = np.random.multinomial(1,[0.1]*10,size=self.batch_size)
            d_loss,_ = sess.run([self.d_loss,self.d_optim],
                                    feed_dict={self.input_i:real_batch,self.input_z:z,self.input_c:c})
            #q_loss,_ = sess.run([self.q_loss,self.q_optim],
            #                        feed_dict={self.input_i:real_batch,self.input_z:z,self.input_c:c})
            for _ in range(ratio):
                z =  np.random.normal(loc=0.0,scale=1.0,size=(self.batch_size,90)) #np.random.uniform(-1,1,(model.batch_size,40))
                g_loss,_ = sess.run([self.g_loss,self.g_optim],feed_dict={self.input_z:z,self.input_c:c})

            d_losses[i]=d_loss
            g_losses[i]=g_loss
            #q_losses[i] =q_loss

        mean_d_loss = np.mean(d_losses).item()
        mean_g_loss = np.mean(g_losses).item()

        img = sess.run(self.sample_outputs, feed_dict={self.input_z:z_sample,self.input_c:c_sample})

        print("time: %4.4f, d_loss: %.8f, g_loss: %.8f "%
              (time.time() - start_time, mean_d_loss, mean_g_loss))

        return img,mean_d_loss,mean_g_loss

if __name__ == "__main__":
    nb_epochs = 200
    cc = CommonChar()
    ic = ImageChar()
    X_all = []
    for c in cc.chars:
        ic.drawText(c)
        X_all.append((ic.toArray()-127.5)/127.5)
    X_train = np.array(X_all)

    if len(X_train.shape)==3:
        X_train = X_train.reshape(X_train.shape + (1,))

    sess = tf.Session(config=config)
    model = Model(batch_size=128,sess=sess)

    d_losses = []
    g_losses = []
    z_sample = np.random.normal(loc=0.0,scale=1.0,size=(model.batch_size,90)) #np.random.uniform(-1, 1, (model.batch_size, 40))
    c_sample = np.zeros((model.batch_size,10))
    for i in range(len(c_sample)):
        c_sample[i,(i//10)%10]=1
    if not os.path.exists("infogan_samples/"):
        os.mkdir("infogan_samples/")

    for epoch in range(nb_epochs):
        print("Epoch is ", epoch)
        img, d_loss, g_loss = model.train_one_epoch(X_train,z_sample, c_sample,sess,ratio=1)
        image = combine_images(img)
        image = image*127.5+127.5
        if len(image.shape)==3:
            image = image[:,:,0]
            Image.fromarray(image.astype(np.uint8)).save("infogan_samples/"+str(epoch)+".png")
        d_losses.append(d_loss)
        g_losses.append(g_loss)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d_losses,label='d_loss')
    ax.plot(g_losses,label='g_loss')
    ax.legend()
    plt.show()
