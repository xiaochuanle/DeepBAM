# 使用说明

## 构建

```bash
# 在3090A解压文件
mkdir BamClass
unzip -d BamClass BamClass.zip
cd BamClass
mkdir build && cd build
cmake ..
make -j 10
```

## 运行

```bash
./BamClass Extract_Feature_and_Call_Mofification \
# pod5 文件路径
/public2/YHC/chr8/pod5_chr8 \
# 按文件名排好序的bam文件地址，不需要跑脚本将bam分开
/public2/YHC/chr8/sorted_by_fn.bam \
# 参考基因组的地址
/public1/OLD_HOME/GPU3090A/ZJY/Human_ref/GCA_000001405.15_GRCh38_major_chr.fa \
# 参考基因组类型
DNA \
# 结果文件路径
/tmp/data/mod_result.txt \
# 模型路径
/public2/YHC/model/traced_script_module_BiLSTM.b51_s15_epoch1_accuracy:0.9883.pt \
# kmer
51 \
# 放到模型里跑的batch_size
4096 \
# 线程池的数量
10 \
# motif
CG \
# 甲基化位点在motif中的位置
0
```

## 输出文件的格式

![image-20230821125848336](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20230821125848336.png)

前三列为位点信息，第四列为甲基化概率，第五列为coverage

