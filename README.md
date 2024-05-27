# Designing-a-Roll-Call-System-with-Facial-Recognition-on-Kubeflow
This project combines classroom roll call system with virtual container technology, presents the whole project in the form of virtual services, and provides web services and model checking functions.

## goals
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

Before training our data, three components are selected as triplets, as shown in **Figure1**, which include an anchor, a positive and a negative from the dataset. Since the model is trained in the Euclidean space, we assume that the distance between two points directly corresponds to the similarity between the two face images. As shown in **Figure2**. after training the model,the distance between the anchor and the positive will be reduced, and that between the anchor and the negative will be increased . If you want to know more detailed FaceNet information, you can refer to [another FaceNet project](https://github.com/mike0355/k8s-facenet-distributed-training).
<div align=center><img width="700" height="250" src="https://user-images.githubusercontent.com/51089749/180136665-e08f777e-2bee-47fa-850e-b22042dbeca5.png"/></div>
<p align ="center"> <b>Figure1. Example of triplet set.</b></p>


<div align=center><img width="600" height="250" src="https://user-images.githubusercontent.com/51089749/137073084-f5c87f57-5eaa-4f83-89c6-cd97408f8a12.png"/></div>
<p align ="center"><b> Figure2. Distance results before and after training.</b></p>

## SVM image classification

In the facial recognition we use the SVM image classification method to implement facial recognition, this model is different from FaceNet model,FaceNet model output is 128-dimensional image feature, and the SVM model output is image class result and score,**Figure3** is SVM schematic.

* **Decision line:** Find a decision line (blue line) in the data that can completely separate the data points, and the sample data must be as far away from the decision line as possible.
* **Decision boundary:** The line connecting the data points closest to the decision line is the decision boundary (purple dashed line).
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

**Tips:** Since this project is built in a virtual container environment, the distributed training part needs to make settings related to network ip. For details, please refer to this [link](https://github.com/mike0355/k8s-facenet-distributed-training/blob/main/step4_Distributed_training.md).

* **Load data:** This pod is responsible for MTCNN face detection for each image data. If no face is detected, the data will be deleted and the face in the next image will continue to be detected. When all the data is detected After the measurement, the face area will be extracted according to the bounding box coordinates output by the face detection, and each face area will be stored in an empty array as an NPZ file.

* **Convert to triplet:** This pod will convert the face data into triples, including Anchor, Positive and Negative data.
* **Distributed training worker1:** This pod will be deployed under node1 and implement distributed training.
* **Distributed training worker2:** This pod will be deployed under node2 and implement distributed training.
* **Distributed training worker3:** This pod will be deployed under node3 and implement distributed training.
* **Feature emb:** This pod is responsible for extracting face feature values from all image data through the FaceNet model, and storing all face features in the NPZ file.

* **SVM training:** This pod mainly uses the facial feature values extracted in the previous work stage as training data to train an SVM image classification model and use it in subsequent applications.

* **Facial recognition:** This pod is responsible for providing web services, uses the SVM model to identify each streamed face image, stores the results in MongoDB, and finally displays the attendee list on the front end of the web page.

* **KServe:** This pod is a KServe model inference server. Users can freely input test data in the external environment to this inference server to make predictions. If the prediction is successful, the recognition results of the test images will be returned.

# How to do?
The development environment of this project is a multi-node Kubernetes cluster environment, and Kubeflow is used to assist in the development of the machine learning pipeline, so this project provides a complete Pipeline, Dataset and Dockerfile.If your development environment has already established a K8s multi-node cluster environment and Kubeflow, you can directly pull our file for use，or if you want to install Kubernetes and deploy kubeflow, you can refer this [link](https://github.com/mike0355/k8s-facenet-distributed-training/blob/main/step1_Local_K8s_and_Kubeflow_setup.md).

1.[Kubeflow pipeline code](https://github.com/mike0355/Designing-a-Roll-Call-System-with-Facial-Recognition-on-Kubeflow/blob/main/Facial-recognition-final-version.ipynb)

2.[Kubeflow pipeline YAML file](https://github.com/mike0355/Designing-a-Roll-Call-System-with-Facial-Recognition-on-Kubeflow/blob/main/Facial-recognition-final-version.yaml)

3.[Dockerfile file](https://github.com/mike0355/Designing-a-Roll-Call-System-with-Facial-Recognition-on-Kubeflow/blob/main/Dockerfile)

## Dataset
This project collects 6,000 face images from ten classmates in the laboratory,as shown as **Figure7**. Under the training set folder and the test set folder, there will be folders of ten classmates, a total of ten categories，as shown as **Figure8**, in the training set data Each category in the folder contains 500 images of younger siblings, and the number of images in each category in the test set folder is 100 images.

<div align=center><img width="500" height="250" src="https://user-images.githubusercontent.com/51089749/180901085-2f496b68-6869-4df0-befd-3c339be161fc.png"/></div>
<p align ="center"> <b>Figure7. Video data of ten younger classmates</b></p>

<div align=center><img width="500" height="250" src="https://user-images.githubusercontent.com/51089749/180901696-784bc4f9-1227-42c7-bc3f-c6eb5e35b438.png"/></div>
<p align ="center"> <b>Figure8. Ten classmates categories </b></p>

## Dockerfile
The project runs in a virtual container environment, and uses the Kubeflow machine learning framework to build a machine learning pipeline. In each stage of the pipeline, an environment that can run the program must be given. Therefore, an environment image file must be created by writing a Dockerfile .

In this Dockerfile, it is mainly to install the packages that this project needs to use when executing the project. OpenCV package is used for image streaming and image processing, pymongo is used for MongoDB database connection and database command operations, Scikit-learn , Keras and Tensorflow are used for machine learning training model related purposes, Flask is used to implement web application suites, and Pillow is used for image processing related operations, such as face area extraction and pixel matrix conversion.**Figure9** is our project Dockerfile detailed content.The image data set collected by this project has been written into the virtual container through dockerfile, so there is no need to additionally collect and read image data when running this project.

<div align=center><img width="500" height="350" src="https://user-images.githubusercontent.com/51089749/180903275-6e39ced4-52cf-44a8-83e7-911feedc917e.png"/></div>
<p align ="center"> <b>Figure9. Dockerfile detailed content </b></p>



## Result: Facial recognition result


**Figure10** is our project facial recognition results. In practical applications, face detection is performed for streaming images. As long as a face is detected, a green bounding box will appear, and the recognition result and confidence level of the face will be displayed. On the contrary, a red bounding box will appear, which is displayed as "unknown".


<div align=center><img width="500" height="350" src="https://user-images.githubusercontent.com/51089749/180903673-902be4dc-a8b9-4ce2-9a62-69f83f1e680c.png"/></div>
<p align ="center"> <b>Figure10. Facial recognition result </b></p>


## Result: MongoDB result
**Figure11** is MongoDB database storage results.When the program performs face recognition, it will send the recognized face results to MongoDB for storage, so as to facilitate the display of the attendee list on the front end of the subsequent webpage.

<div align=center><img width="500" height="350" src="https://user-images.githubusercontent.com/51089749/180904796-0f85a2ae-25ef-4b59-a93e-d190c7054363.png"/></div>
<p align ="center"> <b>Figure11. MongoDB database storage result  </b></p>

## Result: Webpage display
**Figure12** is our project facial recognition webpage.On the home page, there will be a face recognition result area that is responsible for presenting continuous streaming images, and there will be a button object below this area. When user wants to confirm the status of the current attendees, he can press the button below the face recognition area, and when the jump page is triggered, the recognition results stored in the database will be displayed in the form of a list at the front of the webpage,as shown as **Figure13**.


<div align=center><img width="500" height="350" src="https://user-images.githubusercontent.com/51089749/180905482-9caf6592-2323-4faf-b73b-d4ec0531afc4.png"/></div>
<p align ="center"> <b>Figure12. Home webpage </b></p>

<div align=center><img width="1200" height="200" src="https://user-images.githubusercontent.com/51089749/180906088-a8fc4e31-23ce-406f-b45d-12f4aa5c2898.png"/></div>
<p align ="center"> <b>Figure13. Attendee list webpage  </b></p>

## Result: KServe
KServe is a tool for automatic model deployment provided by the Kubeflow community. By deploying the model to the web API provided by Kubeflow to form a model inference server, users can use the python request package to send the prediction data to the HTTP protocol Post action. 

Into the model inference server, a Response will be returned after successful transmission, and the content of the Response is the prediction result after model inference. The input data of KServe in this project is the image data of 10 students, a total of 10 pieces of data. In order to verify whether the returned prediction results are consistent with the images, the 10 image data are used as the English names and student numbers of the 10 students. Make a name and send it to the inference model server to get the return result.
**Figure14** is KServe flow chart，when the predict data is successfully sent to the model server, the log of the model inference server will output messages such as delivery status，and the **Figure15** is KServe result.

<div align=center><img width="700" height="350" src="https://user-images.githubusercontent.com/51089749/180907045-b6d3c833-7eab-4dc2-bfae-3499182c0801.png"/></div>
<p align ="center"> <b>Figure14. KServe flow chart  </b></p>


<div align=center><img width="700" height="350" src="https://user-images.githubusercontent.com/51089749/180908430-6f91aeaf-db1c-4073-9074-44d8831e038e.png"/></div>
<p align ="center"> <b>Figure15. KServe result  </b></p>



















