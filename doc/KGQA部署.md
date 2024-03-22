### java环境安装

下载jdk17 到/ mnt/workspace/java/jdk-21.0.2 下

解压

```shell
tar -zxvf jdk-21_linux-x64_bin.tar.gz
```

配置环境变量

```shell
vim /etc/profile
export PATH="/mnt/workspace/neo4j/neo4j-community-5.16.0/bin:$PATH"

export JAVA_HOME=/mnt/workspace/java/jdk-21.0.2
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH

source /etc/profile
java -version
```

### NEO4J

下载neo4j
到 /mnt/workspace/neo4j/neo4j-community-5.16.0-unix.tar.gz 下

```shell
tar -zxvf neo4j-community-5.16.0-unix.tar.gz
```

修改配置

```shell
vim neo4j.conf
# 修改配置文件
server.default_listen_address=0.0.0.0

server.bolt.enabled=true    # 是否允许Bolt连接，建议设定为true
server.http.enabled=true    # 是否允许http连接，建议设定为true
```

结尾添加

```shell
vim /etc/profile
export PATH="/mnt/workspace/neo4j/neo4j-community-5.16.0/bin:$PATH"
source /etc/profile
```

neo4j 命令

```shell
neo4j --version
neo4j console
neo4j start
```

#### 添加plugins

复制 'apoc-5.16.1-core.jar' 至 /neo4j/plugins
修改：/neo4j/conf/neo4j.conf，新增：
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

#### 修改密码

初始账号密码 neo4j/neo4j

改密码：

```shell
vim conf/neo4j.conf
dbms.security.auth_enabled=false
```

```shell
cypher-shell -d system
# 修改密码
ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO '123456789!';
```

##### 完整profile 文件配置

```shell
export PATH="/mnt/workspace/llm/pkg/neo4j-community-5.16.0/bin:$PATH"
export JAVA_HOME=/mnt/workspace/llm/pkg/jdk-21.0.2
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH
```