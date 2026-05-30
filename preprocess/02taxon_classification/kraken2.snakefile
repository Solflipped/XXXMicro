from os.path import join, abspath, expanduser
import sys
import time


################################################################################
# 指定项目目录 specify project directories
PROJECT_DIR = config["output_directory"]     # 项目流程存放地址
DataBase_DIR = config["database_directory"]  # 各分析工具所需用到的数据库目录
readlength = config['readlength']            # 读段长度

# 将PROJECT_DIR和DataBase_DIR转换成绝对路径 convert PROJECT_DIR and DataBase_DIR to absolute path
if PROJECT_DIR[0] == '~':
    PROJECT_DIR = expanduser(PROJECT_DIR) # 将“～”代表的家目录转换成实际家目录路径
PROJECT_DIR = abspath(PROJECT_DIR) # 返回一个目录的绝对路径
if DataBase_DIR[0] == '~':
    DataBase_DIR = expanduser(DataBase_DIR)
DataBase_DIR = abspath(DataBase_DIR)

# 从样本清单中获取质控后的样本信息 function to get the sample reads from the tsv
def get_sample_reads(sample_file):
    sample_reads = {}  # 样本信息
    paired_end = ''    # 样本是双端读段（True） 还是 单端读段（False）
    with open(sample_file) as sf:
        for l in sf.readlines():
            s = l.strip().split("\t")
            if len(s) == 1 or s[0] == 'Sample' or s[0] == '#Sample' or s[0].startswith('#'):
                continue  # 跳过空行、表头等
            sample = s[0] # 样本名
            # 样本为双端
            if (len(s)==3):
                reads = [s[1],s[2]]
                if paired_end != '' and not paired_end:
                    sys.exit('All samples must be paired or single ended.')
                paired_end = True
            # 样本为单端
            elif len(s)==2:
                reads=s[1]
                if paired_end != '' and paired_end:
                    sys.exit('All samples must be paired or single ended.')
                paired_end = False
            if sample in sample_reads:
                raise ValueError("Non-unique sample encountered!")
            sample_reads[sample] = reads
    return (sample_reads, paired_end)


# 从sample_file中读取样本信息  read in sample info and reads from the sample_file
sample_reads, paired_end = get_sample_reads(config['sample_file'])
# 样本名
sample_names = list(sample_reads.keys())
print("sample_names: ", sample_names)


# 流程： 
# 1、用 kraken2 做物种注释（给 reads 贴 “物种标签”）
# 2、用 bracken 校正丰度（得到真实的物种占比）
# 3、用 KrakenTools 转换结果格式（kreport → mpa 格式，类似 MetaPhlAn 的输出），方便下游合并
# 4、合并所有样本的结果表，得到物种丰度矩阵
# 5、对合并后的结果进行统一样本名处理

################################################################################
rule all:
    input:
        reports = expand(join(PROJECT_DIR,"kraken2/{sample}-kraken2-report.txt"), sample=sample_names),
        brackens = expand(join(PROJECT_DIR,"kraken2/{sample}-bracken-report.txt"), sample=sample_names),
        report_combined = join(PROJECT_DIR,"merged-kraken2-report.txt"),
        bracken_combined =join(PROJECT_DIR,"merged-bracken_new-report.txt"),
        bracken_rel =join(PROJECT_DIR,"merged-bracken_new-report_rel.txt")


################################################################################
##reference:https://lichenhao.netlify.app/post/2020-08-22-krakentools/
##reference: https://hackmd.io/@astrobiomike/kraken2-bracken-standard-build
rule kraken2:
    input:
        r1 = lambda wildcards: sample_reads[wildcards.sample][0],
        r2 = lambda wildcards: sample_reads[wildcards.sample][1],
    output:
        # 统计每个物种的 reads 数量（未校正）
        report = join(PROJECT_DIR, "kraken2", "{sample}-kraken2-report.txt")  
    params:
        db = DataBase_DIR
    threads: 8
    conda: "/data/home/chenliang/apps/miniconda3/envs/kraken2_env"
    shell:"""
        kraken2 --db {params.db} \
        --threads {threads} \
        --report-minimizer-data --minimum-hit-groups 3 \
        --use-names --report {output.report} \
        --paired {input.r1} {input.r2} > /dev/null
    """

################################################################################
rule bracken:
    input:
        report = join(PROJECT_DIR, "kraken2", "{sample}-kraken2-report.txt")
    output:
        brackenout = join(PROJECT_DIR,"kraken2","{sample}-bracken-report.txt"),
        bracken_new = join(PROJECT_DIR,"kraken2","{sample}-bracken_new-report.txt")
    threads: 8
    conda: "/data/home/chenliang/apps/miniconda3/envs/kraken2_env"
    params:
        db = DataBase_DIR,
        readlength = readlength
    shell:"""
        bracken -r {params.readlength} \
        -l S -t 10 \
        -d {params.db} \
        -i {input.report} \
        -o {output.brackenout} \
        -w {output.bracken_new} > /dev/null
    """

################################################################################
rule bracken2mpa:
    input:
        report  = join(PROJECT_DIR,"kraken2","{sample}-kraken2-report.txt"),
        bracken = join(PROJECT_DIR,"kraken2","{sample}-bracken_new-report.txt")
    output:
        report_mpa  = join(PROJECT_DIR,"mpa","{sample}-kraken2-report.txt"),
        bracken_mpa = join(PROJECT_DIR,"mpa","{sample}-bracken_new-report.txt"),
        bracken_rel = join(PROJECT_DIR,"mpa","{sample}-bracken_new-report_rel.txt")
    params:
        readlength=readlength
    conda: "/data/home/chenliang/apps/miniconda3"
    shell:"""
        python /data/home/chenliang/project/PRJCA022804/taxon_classification/scripts/KrakenTools-master/kreport2mpa.py \
        -r {input.report} -o {output.report_mpa}

        python /data/home/chenliang/project/PRJCA022804/taxon_classification/scripts/KrakenTools-master/kreport2mpa.py \
        -r {input.bracken} -o {output.bracken_mpa}

        python /data/home/chenliang/project/PRJCA022804/taxon_classification/scripts/KrakenTools-master/kreport2mpa.py \
        --percentages -r {input.bracken} -o {output.bracken_rel}
    """

################################################################################
rule combine_mpa:
    input:
        report = expand(join(PROJECT_DIR,"mpa","{sample}-kraken2-report.txt"), sample=sample_names),
        bracken = expand(join(PROJECT_DIR,"mpa","{sample}-bracken_new-report.txt"), sample=sample_names),
        bracken_rel = expand(join(PROJECT_DIR,"mpa","{sample}-bracken_new-report_rel.txt"), sample=sample_names)
    output:
        report = join(PROJECT_DIR, "merged-kraken2-report.txt"),
        bracken = join(PROJECT_DIR, "merged-bracken_new-report.txt"),
        bracken_rel = join(PROJECT_DIR,"merged-bracken_new-report_rel.txt")
    params:
        results_dir = join(PROJECT_DIR, "mpa")
    conda: "/data/home/chenliang/apps/miniconda3"
    shell: """
        python /data/home/chenliang/project/PRJCA022804/taxon_classification/scripts/KrakenTools-master/combine_mpa.py\
        -i {params.results_dir}/*-kraken2-report.txt -o {output.report}
        
        python /data/home/chenliang/project/PRJCA022804/taxon_classification/scripts/KrakenTools-master/combine_mpa.py \
        -i {params.results_dir}/*-bracken_new-report.txt -o {output.bracken}
        
        python /data/home/chenliang/project/PRJCA022804/taxon_classification/scripts/KrakenTools-master/combine_mpa.py \
        -i {params.results_dir}/*-bracken_new-report_rel.txt -o {output.bracken_rel}
    """

################################################################################
