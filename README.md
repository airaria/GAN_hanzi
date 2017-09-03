# hanzi
Generate (fake) Chinese characters using GAN  and VAE with tensorflow and keras.

# Results
The resulting images are saved to *_samples/.

##  Training process

12MB gif (tanh in generator):

![Training process](https://github.com/airaria/hanzi/raw/master/process1.gif)


# Some issues

* Using tanh in generator is more stable than Relu, and in some case no BN needed.
* In the generator, relu is much better than tanh in learning strokes.
* Relu is not suitable in the first layer of the generator.
* 判别器成中filter的数目决定了生成的字是否“残”。
