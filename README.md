# Extreme Verification Latency

Python libaray for implementing extreme verification latency algorithms.

To run COMPOSE you will need to instantiate the COMPOSE class: 
reg_compose_QNS3VM = compose.COMPOSE(classifier="QN_S3VM", method="a_shape", verbose = 2,num_cores=0.8, selected_dataset='1CHT')
reg_compose_QNS3VM.run()

The following parameters need to be passed in: 
1. classifier : QN_S3VM or label_propagation 
2. method : type of clustering 'gmm' or 'a_shape'
    - gmm accounts for Fast COMPOSE
    - a_shape accounts for alpha shapes
3. verbose : 0: doesnt print out , 1 prints to command line 
4. num_cores: percent in which you wish to operate 0.8 = 80% of available cores 
5. selected dataset : options available :
    ['UG_2C_2D','MG_2C_2D','1CDT', '2CDT', 'UG_2C_3D','1CHT','2CHT','4CR','4CREV1','4CREV2','5CVT','1CSURR', '4CE1CF','FG_2C_2D','GEARS_2C_2D', 'keystroke', 'UG_2C_5D', 'UnitTest']