# echoGAN
Code for paper echoGAN: Extending the Field of View in Transthoracic Echocardiography Through Conditional GAN-Based Outpainting

Authored by:
Matej Gazda, Jakub Gazda, Robert Kanasz, Samuel Kadoury, Peter Drotar

## Preprocessing steps:
Scripts to run:
preprocess\_camus.py --src DIR\_TO\_NIFTI\_CAMUS\_DATASET --angle\_to\_cut 30 --dst /tmp/camus\_cut\_30

Output:
Each image will contain the following:
\_whole img: full img
\_red: inner part of the image (context)
\_diff outer part of the image

also .npy files for mask (0/1) for outpainting region


