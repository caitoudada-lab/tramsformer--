##########Redis事务#############

    与关系型数据库不同，不保证全部成功

保证

    1.发送EXEC命令之前，所有命令放入队列中缓存起来，不立即执行

    2.收到EXEC命令之后，事务开始执行，即使其中任何一个命令执行失败，其他命令仍然会执行

    3.命令执行过程中，其他客户端请求的命令，不会插入到事务的执行命令序列中。

开启一个事务

    MULTI
    执行后，进入TX模式



##########Redis持久化#############

重启断电后，数据依然在

持久化两种方式

    RDB (Redis Database)

    指定时间间隔内，将内存中的数据快照写入磁盘
    使用配置文件自动或者save命令手动
    更适合做备份

    AOF (Append Only File)


bgsave命令，单独创建一个子进程，将内存中的数据写入硬盘，但又性能损耗，处理过程中，Redis不能做任何请求，没有秒级的快照

    AOF:

    追加文件
    执行写命令时，不仅将命令写入内存，还会写入到一个追加的文件中，即AOF文件。
    重启之后，就重新执行AOF中的命令，重新构建数据库的内容

    开启方式：配置文件中 appendonly 值改成yes



##########Redis主从复制#############

    将一台Redis服务器，复制到其他Redis服务器。
    即主结点（master），从节点(slave)
    主节点唯一，从节点可多个

数据的复制是单向的：主到从，异步的方式 

一般主节点：写操作
    从节点：读操作



################Redis哨兵模式###############

主节点宕机了，还是要手动将一台从节点提升为主节点
需要人工干预

哨兵模式 会以一个独立的进程，运行在Redis集群中，监控各个节点

几个功能：

    1.监控
        不断发送命令
    2.通知
        如果有节点有问题，会通过发布订阅模式（联想 Kafka），通知其他节点
    3.自动故障转移
        主节点不能自动工作的时候，开始自动故障转移，将一台从节点提升为主节点，再将其他从节点指向新的主节点。

配置文件：sentinel.conf

    sentinel monitor 主节点名字 ip port 1

    1 表示只需要一个哨兵节点同意，就可以故障转移了
启动：

    redis-sentinel  sentinel.conf

    哨兵本身也是一个进程，生产环境中一般使用3个保证高可用，这三个中会选出一个领导者，监控其他节点。
    领导者挂了，会重新选一个领导者。