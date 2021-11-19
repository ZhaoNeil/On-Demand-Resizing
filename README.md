# On-Demand-Resizing

Designed a mechanism for on-demand resource resizing of existing containers.

**data** file contains analysis code for the performance of pods.

**pod-validation** file contains the deployment scripts for the pod validation on minikube.

**recommender** file is the code for vertical pod autoscaling recommender component. Modified from this [code](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler/pkg/recommender). Replace it in *autoscaler/vertical-pod-autoscaler/pkg/recommender/* to generate a new image. Change the image name to new image name instead of the default one in *autoscaler/vertical-pod-autoscaler/deploy/recommender-deployment.yaml*

**redis** file is the deployment scripts for the Redis and YCSB.
