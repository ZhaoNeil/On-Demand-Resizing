#! /bin/bash

no_files=300

for ((i=0; i<$no_files; i ++))
do

kubectl delete -f ./yaml/test_file_${i}.yaml
kubectl delete -f ./yaml/test_vpa_${i}.yaml

rm ./yaml/test_file_${i}.yaml
rm ./yaml/test_vpa_${i}.yaml

done
