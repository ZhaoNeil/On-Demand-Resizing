kubectl delete -f redis-statefulset.yaml
kubectl delete -f ycsb.yaml
kubectl delete -f redis-vpa.yaml
cd ../autoscaler/vertical-pod-autoscaler/hack && ./vpa-down.sh
