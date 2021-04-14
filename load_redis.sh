kubectl exec -it ycsb -- sh -c "cd YCSB && ./bin/ycsb load redis -s -P workloads/workloada -p recordcount=5000000 -p operationcount=5000000 -p "redis.host=172.17.0.4" -p "redis.port=6379""
