#! /bin/bash

no_files=300

for ((i=0; i<$no_files; i ++))
do

# create the yaml files
cat > ./yaml/test_file_${i}.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sleep${i}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sleep${i}
  template:
    metadata:
      labels:
        app: sleep${i}
    spec:
      containers:
      - name: sleep${i}
        image: zhaoneil/sleep_container:var_workload
        resources:
          limits:
            memory: 50Mi
          requests:
            memory: 50Mi

EOF

# create the vpa files
#
#
cat > ./yaml/test_vpa_${i}.yaml << EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: sleep-vpa${i}
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind:       Deployment
    name:       sleep${i}
  updatePolicy:
    updateMode: "Off"
EOF
kubectl apply -f ./yaml/test_file_${i}.yaml
kubectl apply -f ./yaml/test_vpa_${i}.yaml

done
