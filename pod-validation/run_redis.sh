kubectl exec -it ycsb -- sh -c "cd YCSB && ./bin/ycsb run redis -s -P workloads/workloada -p recordcount=2500000 -p operationcount=2500000 -p "redis.host=172.17.0.3" -p "redis.port=6379""
