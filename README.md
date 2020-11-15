# MobileLearning

## Example project to capture and process Mobile Data in a cloud native MachineLearning pipeline

### Architecture
![Architecture](https://raw.githubusercontent.com/juergen1976/MobileLearning/master/images/AirflowArchitecture.jpg)

The architecture above with Zookeeper/Kafka/MLFlow and Airflow realizes a ML pipeline workflow to process and
use mobile movement data.

The sample mobile hybrig flutter app is from my repository SmartMovement.

### The mobile app
![Architecture](https://raw.githubusercontent.com/juergen1976/MobileLearning/master/images/SmartMovementFlutterApp.jpg)

### This Repository contains follwing modules
+ Some sample movement data in the `data` directory
+ Keras initial training code to build a simple DL network to predict device Lock/Unlocking in `dags\src`
+ Airflow Workflow files in `dags`
+ Docker files to compose the pipline architecture with Kafka/Zookeeper/MLFlow/Airflow in `docker`
+ Docker compose file
+ Serialized model files in `dags\models` uses in the ML pipeline between components

