Convolution (Conv2d): The mathematical operation of sliding a small matrix (the filter/kernel) over an input image to produce a feature map.

Kernel / Filter: The small weight matrix (usually 3x3 or 5x5) that slides across the image. The network learns the numbers inside this kernel during backpropagation to detect useful features.

Stride: How many pixels the filter moves at a time as it scans the image. A stride of 1 moves pixel by pixel.

Padding: Adding extra blank pixels (usually zeros) around the border of the original image so the filter can properly scan the very edges.

Max Pooling (MaxPool2d): A downsampling operation that slides a window over the feature map and only keeps the maximum value in that window, throwing away the rest to reduce data size.

Flatten: The process of taking the final 2D feature maps and unrolling them into a 1D list so they can be fed into a standard Chief (Fully Connected Layer) to make the final diagnosis.