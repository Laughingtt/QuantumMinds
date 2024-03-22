NEO4J:
下载neo4j
到 /mnt/workspace/neo4j/neo4j-community-5.16.0-unix.tar.gz 下

tar -zxvf neo4j-community-5.16.0-unix.tar.gz

vim neo4j.conf
修改配置文件
server.default_listen_address=0.0.0.0

server.bolt.enabled=true    # 是否允许Bolt连接，建议设定为true
server.http.enabled=true    # 是否允许http连接，建议设定为true


vim /etc/profile

结尾添加
export PATH="/mnt/workspace/neo4j/neo4j-community-5.16.0/bin:$PATH"

. /etc/profile   （点 后面有空格）

neo4j --version
neo4j console

复制 'apoc-5.16.1-core.jar' 至 /neo4j/plugins
修改：/neo4j/conf/neo4j.conf，新增：
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

改密码：
neo4j/conf/neo4j-sever.properties
dbms.security.auth_enabled=true
cypher-shell     /bin/cypher-shell -d system
:exit

ALTER USER neo4j SET PASSWORD 'mynewpass';
ALTER USER neo4j SET PASSWORD '123456789a';
ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO '123456789!';



Java
下载jdk17
到/ mnt/workspace/java/jdk-21.0.2 下

/mnt/workspace/neo4j/neo4j-community-5.16.0/bin
export PATH="/mnt/workspace/neo4j/neo4j-community-5.16.0/bin:$PATH"


tar -zxvf jdk-21_linux-x64_bin.tar.gz

vim /etc/profile

结尾添加
export JAVA_HOME=/mnt/workspace/java/jdk-21.0.2 #你自己的安装路径
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH

. /etc/profile   （点 后面有空格）
java -version



export JAVA_HOME=/mnt/workspace/llm/pkg/jdk-21.0.2 
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH