sh run_redis.sh & run_redis=$!
sh monitor.sh & monitor=$!
wait $run_redis
kill $monitor
