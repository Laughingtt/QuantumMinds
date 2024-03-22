echo "start neo4j"
neo4j --version
neo4j start >> neo4j.log
echo "start app_cuda"
nohup python app_cuda.py  >> api.log &
echo "start finished"