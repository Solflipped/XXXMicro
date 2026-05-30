from os.path import join, abspath, expanduser
import sys
import time

################################################################################
# 指定项目目录 specify project directories
PROJECT_DIR = config["output_directory"]     # 项目流程存放地址
# 读段所在目录
INPUT_DIR = config["input_directory"]

# 将PROJECT_DIR和DataBase_DIR转换成绝对路径 convert PROJECT_DIR and DataBase_DIR to absolute path
if PROJECT_DIR[0] == '~':
    PROJECT_DIR = expanduser(PROJECT_DIR) # 将“～”代表的家目录转换成实际家目录路径
PROJECT_DIR = abspath(PROJECT_DIR) # 返回一个目录的绝对路径
if INPUT_DIR[0] == '~':
    INPUT_DIR = expanduser(INPUT_DIR) # 将“～”代表的家目录转换成实际家目录路径
INPUT_DIR = abspath(INPUT_DIR) # 返回一个目录的绝对路径

# ARGs-OAP数据库路径
ARGS_OAP_DB = "/data/home/chenliang/apps/miniconda3/envs/args_oap_env/lib/python3.13/site-packages/args_oap/db/sarg.fasta.dmnd"


################################################################################
rule all:
    input:
        gene = join(PROJECT_DIR,"args_oap","normalized_cell.gene.txt"),
        type = join(PROJECT_DIR,"args_oap","normalized_cell.type.txt"),
        subtype = join(PROJECT_DIR,"args_oap","normalized_cell.subtype.txt")
        
################################################################################
rule args_oap_one:
    output:
        metadata = join(PROJECT_DIR, "args_oap","metadata.txt"),
        extracted_fa = join(PROJECT_DIR, "args_oap","extracted.fa")  # 添加extracted.fa作为输出
    params:
        indir = INPUT_DIR,
        outdir = join(PROJECT_DIR,  "args_oap")
    conda:"/data/home/chenliang/apps/miniconda3/envs/args_oap_env"
    threads: 8
    shell:"""
        mkdir -p {params.outdir}
        args_oap stage_one -i {params.indir} -o {params.outdir} -f fq.gz -t {threads}
    """


################################################################################
rule blastx:
    input:
        extracted_fa = rules.args_oap_one.output.extracted_fa
    output:
        blastout = join(PROJECT_DIR, "args_oap", "blastout.txt")
    params:
        db = ARGS_OAP_DB
    conda:"/data/home/chenliang/apps/miniconda3/envs/args_oap_env"
    threads: 4  # 使用单线程避免线程错误
    shell:"""
        diamond blastx -d {params.db} \
               -q {input.extracted_fa} \
               -o {output.blastout} \
               -f 6 qseqid sseqid pident length qlen slen evalue bitscore \
               -e 1e-07 \
               -k 5 \
               -p {threads}
    """
    
################################################################################
rule args_oap_two:
    input:
        metadata = rules.args_oap_one.output.metadata,
        extracted_fa = rules.args_oap_one.output.extracted_fa,
        blastout = rules.blastx.output.blastout  # 添加blast输出作为输入
    output:
        gene = join(PROJECT_DIR,"args_oap","normalized_cell.gene.txt"),
        type = join(PROJECT_DIR,"args_oap","normalized_cell.type.txt"),
        subtype = join(PROJECT_DIR,"args_oap","normalized_cell.subtype.txt")
    params:
        indir = join(PROJECT_DIR,  "args_oap")
    conda:"/data/home/chenliang/apps/miniconda3/envs/args_oap_env"
    threads: 8  # 减少线程数
    shell:"""
        args_oap stage_two -i {params.indir} -t {threads} --blastout {input.blastout}
    """
################################################################################