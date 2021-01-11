To install hdfs on a new node:

(1) add the node to the slaves file;

(2) install java-oracle-8 on the new node (https://askubuntu.com/questions/521145/how-to-install-oracle-java-on-ubuntu-14-04);

(3) zip /usr/local/hadoop on the master node and copy it to the new node ( /usr/local/hadoop.tar    cmd:tar zcvf hadoop.tar /usr/local/hadoop);

(4) set hadoop path in .bashrc if necessary (echo "export HADOOP_HOME=/usr/local/hadoop" | tee -a ~/.bashrc; echo "export PATH=\$PATH:\$HADOOP_HOME/bin" | tee -a ~/.bashrc;  source ~/.bashrc);

(5) mount another disk if necessary, create /data/hadoop

(6) start the node without the need of stopping the whole cluster;

(7) visit master:50070 to check.


To access hadoop yarn resource manager, visit http://master:8089

To compile and run job, visit https://hadoop.apache.org/docs/r2.8.0/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html#Example:_WordCount_v1.0

All logs are in /usr/local/hadoop/logs/; specifically, application logs are in /usr/local/hadoop/logs/userlogs

job history: $HADOOP_HOME/sbin/mr-jobhistory-daemon.sh start historyserver
then visit master:50030 (default 19888)

