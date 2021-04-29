while true
do
	kubectl top pod | grep redis-master-0 >> data_run.txt
	kubectl top pod | grep redis-worker-0 >> data_run.txt
	kubectl top pod | grep ycsb >> data_run.txt
	sleep 1
done
