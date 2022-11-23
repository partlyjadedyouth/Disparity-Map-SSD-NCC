# Disparity-Map-SSD-NCC
Compute disparity maps by sum of spatial difference and normalized cross correlation

## Brief Explanations
### SSD
L1.png and R1.png are the rectified images with the same brightness.

We can get a disparity map by using SSD in this case. 

The corresponding output is 'disparity_c_L1R1.png'.

### NCC
R2.png is an image that is the same as R1.png, except for the brightness.

If we try to get a disparity map between L1.png and R2.png by SSD, we get an output that is totally unrecognizable.

SSD is not robust to the brightness. In this case, we should use NCC.

The corresponding output is 'disparity_d_L1R2.png'.

If we use the same function to L1.png and R1.png, we get 'disparity_d_L1R1.png'. 

As you can see, the two outputs are almost identical.
