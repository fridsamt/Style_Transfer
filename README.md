# Style_Transfer
First, you need to download VGG model to ./model from following link:
http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

Second, put content image and style images seprately to ./content_images and ./style_images. Then you should change images' filename in the config.cfg. If you want to use mutiple style images, make sure change 'style_weights' in config.cfg so the number of weights is consistent with the number of style images.

Now you can run python style_transfer.py in console and wait for the output. By default the output will be ./output/output.png

The time for generating a single images is around 20 miniutes on GTX1080.

Feel free to change other parameters in config.cfg as long as you know their function.
