import numpy as np
import time
from PIL import Image
import math,os
import matplotlib.pyplot as plt
import tensorflow as tf
from Chinese_inputs import CommonChar, ImageChar

z_dim = 32
batch_size = 128

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

class VAE_AE():
    def encoder(self,x,is_training, is_reuse):
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        def lrelu(x, leak=0.2, name="lrelu"):
            return tf.maximum(x, leak * x,name=name)

        with tf.variable_scope("encoder",reuse=is_reuse) as scope:
            with tf.variable_scope("conv1"):
                outputs = tf.layers.conv2d(inputs=x,filters=64,
                                           kernel_size=(5,5),strides=(2,2),
                                           padding='SAME',activation=None,kernel_initializer=w_init
                                           )
                outputs = lrelu(outputs)
            with tf.variable_scope("conv2"):
                outputs = tf.layers.conv2d(inputs=outputs,filters=128,
                                           kernel_size=(5,5),strides=(2,2),
                                           padding='SAME',activation=None,kernel_initializer=w_init
                                           )
                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=gamma_init)
                outputs2 = lrelu(outputs)
            with tf.variable_scope("conv3_log_Sigma"):
                outputs = tf.layers.conv2d(inputs=outputs2,filters=256,
                                           kernel_size=(5,5),strides=(2,2),
                                           padding='SAME',activation=None,kernel_initializer=w_init
                                           )
                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=gamma_init)
                outputs = lrelu(outputs)
                outputs = tf.reshape(outputs,[outputs.get_shape()[0].value,-1])
                log_Sigma = tf.layers.dense(outputs,z_dim)
            with tf.variable_scope("conv3_mu"):
                outputs = tf.layers.conv2d(inputs=outputs2, filters=256,
                                           kernel_size=(5, 5), strides=(2, 2),
                                           padding='SAME', activation=None, kernel_initializer=w_init
                                           )
                #outputs = tf.layers.batch_normalization(outputs, training=is_training, gamma_initializer=gamma_init)
                outputs = lrelu(outputs)
                outputs = tf.reshape(outputs, [outputs.get_shape()[0].value, -1])
                mu = tf.layers.dense(outputs,z_dim)
            return mu,log_Sigma

    def decoder(self,e,is_training, is_reuse):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)
        with tf.variable_scope("decoder",reuse=is_reuse) as scope:
            with tf.variable_scope("h0"):
                outputs = tf.layers.dense(e,512*4*4,kernel_initializer=w_init)
                outputs = tf.reshape(outputs,[-1,4,4,512])
                outputs = tf.layers.batch_normalization(outputs, training=is_training, gamma_initializer=g_init)
                outputs = tf.nn.tanh(outputs)
            with tf.variable_scope("conv1"):
                outputs = tf.layers.conv2d_transpose(outputs,256,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
                outputs = tf.nn.relu(outputs)
            with tf.variable_scope("conv2"):
                outputs = tf.layers.conv2d_transpose(outputs,128,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
                outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
                outputs = tf.nn.relu(outputs)

            with tf.variable_scope("conv3"):
                outputs = tf.layers.conv2d_transpose(outputs,64,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
                #outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
            with tf.variable_scope("outputs"):
                logits = tf.layers.conv2d_transpose(outputs, 1, (5, 5), (2, 2), 'same', activation=None,kernel_initializer=w_init)
                outputs = tf.nn.sigmoid(logits)
            return outputs,logits

    def autoencoder(self,x,is_training,is_reuse):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)
        with tf.variable_scope("autoencoder",reuse=is_reuse) as scope:
            outputs = tf.layers.conv2d(x,32,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d(outputs,32,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d(outputs,16,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            outputs = tf.nn.relu(outputs)

            outputs = tf.layers.conv2d_transpose(outputs,32,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d_transpose(outputs,32,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            outputs = tf.nn.relu(outputs)
            logits = tf.layers.conv2d_transpose(outputs,1,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            outputs = tf.nn.sigmoid(logits)
        return outputs,logits

    def ae_loss(self,logits,labels):
        logits = tf.reshape(logits, [logits.get_shape()[0].value, -1])
        labels = tf.reshape(labels, [labels.get_shape()[0].value, -1])
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))

    def vae_loss(self,log_Sigma,mu,logits):
        KL_loss = 0.5*tf.reduce_sum(tf.exp(log_Sigma)+tf.square(mu)-1.-log_Sigma,axis=1)
        logits = tf.reshape(logits,[logits.get_shape()[0].value,-1])
        x_input = tf.reshape(self.x_input,[self.x_input.get_shape()[0].value,-1])
        rec_loss= tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=x_input),axis=1)
        return tf.reduce_mean(KL_loss + rec_loss)


    def __init__(self,batch_size,sess):
        self.batch_size = batch_size
        self.x_input = tf.placeholder(tf.float32,shape=(batch_size,64,64,1))
        self.z = tf.placeholder(tf.float32,shape=(batch_size,z_dim))
        self.e = tf.random_normal(shape=(batch_size,z_dim))


        mu,log_Sigma = self.encoder(self.x_input,is_training=True, is_reuse=False)
        self.reconstruct,recon_logits = self.decoder(mu+self.e*tf.exp(log_Sigma/2),is_training=True,is_reuse=False)
        self.denoised,   denoi_logits = self.autoencoder(self.reconstruct,is_training=True,is_reuse=False)


        self.infer, _ = self.decoder(self.z,is_training=False, is_reuse=True)
        self.denoi_infer, _ = self.autoencoder(self.infer,is_training=False,is_reuse=True)

        self.loss1 = self.vae_loss(log_Sigma,mu,recon_logits)
        self.loss2 = self.ae_loss(denoi_logits,self.reconstruct)

        vae_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(vae_update):
            self.opt_vae = tf.train.AdamOptimizer().minimize(self.loss1)
        self.opt_ae = tf.train.AdamOptimizer().minimize(self.loss2,var_list=[var for var in tf.trainable_variables() if 'autoencoder' in var.name])
        sess.run(tf.global_variables_initializer())

    def train_one_epoch(self, real_images, sess):
        shuffled_images = real_images[np.random.permutation(len(real_images))]

        nb_batch = len(real_images)//self.batch_size
        losses = np.zeros(nb_batch)

        start_time = time.time()
        for i in range(nb_batch):
            if i<10:
                real_batch = shuffled_images[i*self.batch_size:(i+1)*self.batch_size]
                loss1,_ = sess.run([self.loss1,self.opt_vae], feed_dict={self.x_input:real_batch})
                losses[i]=loss1
            else:
                real_batch = shuffled_images[i*self.batch_size:(i+1)*self.batch_size]
                #loss1,_ ,_= sess.run([self.loss1,self.opt_vae,self.opt_ae], feed_dict={self.x_input:real_batch})
                loss1,_ = sess.run([self.loss1, self.opt_vae], feed_dict={self.x_input: real_batch})
                losses[i]=loss1

        mean_loss = np.mean(losses).item()

        print("time: %4.4f, loss: %.8f" % (time.time() - start_time, mean_loss))

        return mean_loss

    def inference(self,z,sess):
        #generated_images1,generated_images2 = sess.run([self.infer,self.denoi_infer],feed_dict={self.z:z})
        generated_images1 = sess.run(self.infer, feed_dict={self.z: z})
        return generated_images1

if __name__ == '__main__':
    nb_epochs = 500
    cc = CommonChar()
    ic = ImageChar()
    X_all = []
    for c in cc.chars:
        ic.drawText(c)
        X_all.append((ic.toArray()/255.))
    X_train = np.array(X_all)
    if len(X_train.shape)==3:
        X_train = X_train.reshape(X_train.shape + (1,))

    sess = tf.Session()
    model = VAE_AE(batch_size,sess)
    losses = []
    z_sample = np.random.normal(loc=0.0,scale=1.0,size=(model.batch_size,z_dim))
    #z_sample = np.array([[(i,j) for i in np.linspace(-2,2,10)] for j in np.linspace(-2,2,13)]).reshape(130,2)[:batch_size]
    if not os.path.exists("vae_samples/"):
        os.mkdir("vae_samples/")

    for epoch in range(1,nb_epochs+1):
        print("Epoch [{} / {}] ".format(epoch,nb_epochs))
        loss = model.train_one_epoch(X_train,sess)
        if epoch%2==0:
            #img,denoi_img = model.inference(z_sample,sess)
            img = model.inference(z_sample, sess)

            image = combine_images(img) #(W*H*D)
            image = image*255
            if image.shape[-1]==1:
                image = image[:,:,0]
                Image.fromarray(image.astype(np.uint8)).save("vae_samples/"+str(epoch)+".png")
            '''
            image = combine_images(denoi_img) #(W*H*D)
            image = image*255
            if image.shape[-1]==1:
                image = image[:,:,0]
                Image.fromarray(image.astype(np.uint8)).save("vae_samples/"+str(epoch)+"_denoised.png")
            '''
        losses.append(loss)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses,label='loss')
    ax.legend()
    plt.show()