# On-Demand-Resizing

Design a mechanism and policies for on-demand (resource) resizing of existing containers.

**data** file contains analysis code for the performance of pods.

**pod-validation** file contains the deployment scripts for the pod validation on minikube.

**recommender** file is the code for vertical pod autoscaling recommender component. Modified from this [code](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler/pkg/recommender). Replace it in *autoscaler/vertical-pod-autoscaler/pkg/recommender/* to generate a new image.

**redis** file is the deployment scripts for the Redis and YCSB.
