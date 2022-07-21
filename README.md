# Designing-a-Roll-Call-System-with-Facial-Recognition-on-Kubeflow
This project combines classroom roll call system with virtual container technology, presents the whole project in the form of virtual services, and provides web services and model checking functions.

## Goals
This project will implement the following items: 

* A classroom roll call system combining MTCNN face detection and face recognition methods.
* Use FaceNet to implement facial feature extraction.
* Use Tensorflow MuiltiWorkerMirroredStrategy to demonstrate distributed training.
* Use No.SQL mongoDB to save our recognition result.
* The process of face detection and face recognition is built into a complete machine learning workflow through Kubeflow. 
* Use Flask+HTML to implement web front-end web design.
* Model testing and monitoring models using KServe.

## MTCNN face detection
  MTCNN English full name is Multi-task Cascaded Convolutional Networks, a multi-task cascaded convolutional neural network composed of three network architectures: P-Net (proposal network), R-Net (refine network) and O-Net (output network) network.
    
  The significance of cascading is that the output of the previous layer of network architecture will be the input of the next layer of network architecture, and the network architecture is mainly used in face detection and face key point detection in the field of computer vision.

## FaceNet

In this project, a algorithm called FaceNet is used to implement a face recognition on K8s. FaceNet presents a algorithm to train the features of Euclidean as a similarity between two face images, and output the distance as the similarity between two face images. In addition, we use triplet loss function to optimized the model.  
Before training our data, three components are selected as triplets, as shown in **Figure1**, which include an anchor, a positive and a negative from the dataset. Since the model is trained in the Euclidean space, we assume that the distance between two points directly corresponds to the similarity between the two face images. As shown in **Figure2**. after training the model,the distance between the anchor and the positive will be reduced, and that between the anchor and the negative will be increased . 
<div align=center><img width="700" height="250" src="https://user-images.githubusercontent.com/51089749/180136665-e08f777e-2bee-47fa-850e-b22042dbeca5.png"/></div>
<p align ="center"> <b>Figure1. Example of triplet set.</b></p>


<div align=center><img width="600" height="250" src="https://user-images.githubusercontent.com/51089749/137073084-f5c87f57-5eaa-4f83-89c6-cd97408f8a12.png"/></div>
<p align ="center"><b> Figure2. Distance results before and after training.</b></p>

## SVM image classification

In the facial recognition we use the SVM image classification method to implement facial recognition, this model is different from FaceNet model,FaceNet model output is 128-dimensional image feature, and the SVM model output is image class result and score,**Figure3** is SVM schematic.

* Decision line: Find a decision line (blue line) in the data that can completely separate the data points, and the sample data must be as far away from the decision line as possible.
* Decision boundary: The line connecting the data points closest to the decision line is the decision boundary (purple dashed line).
<div align=center><img width="400" height="250" src="https://user-images.githubusercontent.com/51089749/180137944-4ed15fa3-3f8c-4803-9452-c03dd2cd7b4b.png"/></div>
<p align ="center"> <b>Figure3. SVM　schematic</b></p>

## MultiWorkerMirroredStrategy

In this project, we use the tensorflow MultiWorkerMirroredStrategy to impplement three workers distributed training,**Figure4** is the Ring All Reduce distributed training method schematic.In this algorithm, we spliced all training workers into a ring, and each worker is responsible for sending and receiving parameters. This method can greatly reduce the time when training a large number of resources.
<div align=center><img width="300" height="250" src="https://user-images.githubusercontent.com/51089749/180139567-878daec0-cefb-4798-a6d8-a2f0ec793665.png"/></div>
<p align ="center"> <b>Figure4. Ring All reduce</b></p>

When the distributed training is over, each worker responsible for training will have a complete  model. As shown as **Figure5** .
<div align=center><img width="300" height="250" src="https://user-images.githubusercontent.com/51089749/180159861-3085aa1a-fafd-4409-a81b-a4cac22450c5.png"/></div>
<p align ="center"> <b>Figure5. After distributing training schematic</b></p>

## Kubeflow pipeline
The Kubeflow pipeline of this project is shown in the **figure6**. The overall process includes face detection, triplet conversion, distributed training, feature value extraction, SVM model training, and finally web serving and KServe model monitoring and testing.

<div align=center><img width="500" height="550" src="https://user-images.githubusercontent.com/51089749/180161740-25a2ea4e-f936-41eb-8aeb-f2fbb3ca6255.png"/></div>
<p align ="center"> <b>Figure6. Roll call system kubeflow pipeline</b></p>

At this part we will explain the function of each pods.

* **face-detect-pvc:** This pod responsible for providing a common storage space for the pipeline, which can be used to store training files, image data files or weight files.

* **Load data:** This pod is responsible for MTCNN face detection for each image data. If no face is detected, the data will be deleted and the face in the next image will continue to be detected. When all the data is detected After the measurement, the face area will be extracted according to the bounding box coordinates output by the face detection, and each face area will be stored in an empty array as an NPZ file.

* **Convert to  triplet:** This pod will convert the face data into triples, including Anchor, Positive and Negative data.

* **Distributed training worker1:** This pod will be deployed under node1 and implement distributed training.
* **Distributed training worker2:** This pod will be deployed under node2 and implement distributed training.
* **Distributed training worker3:** This pod will be deployed under node3 and implement distributed training.
* **Feature emb:** This pod is responsible for extracting face feature values from all image data through the FaceNet model, and storing all face features in the NPZ file.

* **SVM training:** This pod mainly uses the facial feature values extracted in the previous work stage as training data to train an SVM image classification model and use it in subsequent applications.

* **Facial recognition:** This pod is responsible for providing web services, uses the SVM model to identify each streamed face image, stores the results in MongoDB, and finally displays the attendee list on the front end of the web page.

* **KServe:** This pod is a KServe model inference server. Users can freely input test data in the external environment to this inference server to make predictions. If the prediction is successful, the recognition results of the test images will be returned.







