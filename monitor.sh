while true
do
	kubectl top pod | grep redis-0 >> data1.txt
	kubectl top pod | grep redis-1 >> data1.txt
	kubectl top pod | grep ycsb >> data1.txt
	sleep 1
done
