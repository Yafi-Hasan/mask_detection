# Mask-Usage Detection

## Description
The goal of this project is to classify medical-mask-usage in three different classes.
- Not wearing a mask
- Properly wearing a mask
- Not properly wearing a mask (only covered the mouth, not the nose)

The classification algorithm used in this project is customize MobileNet-V2 computer vision, machine learning, architecture. 2340 images used for training and testing process (780 images for each class). 85% images used for training and 15% images used for testing. The images used to train the model comes from three different datasets. Those datasets are:
- MaskedFace-Net Dataset (Cabani et al., 2021)
- Medical Mask Dataset (Humans in the loop, 2021)
- Real-Time-Medical-Mask-Detection Dataset (Nagrath et al., 2020)

The new customized dataset created from those datasets can be accessed in https://drive.google.com/drive/folders/1zZV55gzvXXtr3Fu8QTMkJWD-ZOWasbHW?usp=sharing


## Dataset Reference
- Cabani, A., Hammoudi, K., & Benhabiles, H. (2021). MaskedFace-Net-A Dataset of Correctly/Incorrectly Masked Face Images in The Context of Covid-19. Smart Health, 19(January).
- Humans in the Loop. (2021). Medical Mask Dataset. Humans in the Loop. https://humansintheloop.org/resources/datasets/medical-mask-dataset/
- Nagrath, P., Jain, R., Madan, A., Arora, R., Kataria, P., & Hemanth, J. (2020). SSDMNV2: A Real time DNN-Based Face Mask Detection System using Single Shot Multibox Detector and MobileNetV2. Sustainable Cities and Society, 66(August 2020), 102692. https://doi.org/10.1016/j.scs.2020.102692
