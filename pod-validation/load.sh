sh load_redis.sh & load_redis=$!
sh monitor.sh & monitor=$!
wait $load_redis
kill $monitor
