from os.path import join, abspath, expanduser
import sys
import time


################################################################################
# 指定项目目录 specify project directories
PROJECT_DIR = config["output_directory"]     # 项目流程存放地址

# 将PROJECT_DIR和DataBase_DIR转换成绝对路径 convert PROJECT_DIR and DataBase_DIR to absolute path
if PROJECT_DIR[0] == '~':
    PROJECT_DIR = expanduser(PROJECT_DIR) # 将“～”代表的家目录转换成实际家目录路径
PROJECT_DIR = abspath(PROJECT_DIR) # 返回一个目录的绝对路径


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
# 1. concat: 双端样本合并(如为双端测序)
# 2. humann3: 使用 humann3 + metaphlan4 进行功能分析
# 3. merge: 合并样本的结果
# 4. clean: 清理临时文件

################################################################################
rule all:
    input:
        join(PROJECT_DIR, "taxonomy.tsv"),            # 物种分类结果
        join(PROJECT_DIR, "pathabundance.tsv"),       # 通路丰度结果
        join(PROJECT_DIR, "pathabundance_relab.tsv"), # 相对丰度通路结果
        join(PROJECT_DIR, "cleaned")


################################################################################
rule concat:  # 双端合并为单个文件
    input:
        r1 = lambda wildcards: sample_reads[wildcards.sample][0],
        r2 = lambda wildcards: sample_reads[wildcards.sample][1]
    output:
        clean_read = join(PROJECT_DIR, "temp", "{sample}_clean.fq.gz")  
    params:
        outdir = join(PROJECT_DIR,  "temp")
    conda: "/data/home/chenliang/apps/miniconda3"
    shell:"""
        mkdir -p {params.outdir}
        cat {input.r1} {input.r2} > {output.clean_read}
    """

################################################################################
rule humann3:
    input:
        clean_read = rules.concat.output.clean_read
    output:
        genefamilies = join(PROJECT_DIR,  "humann3", "{sample}_genefamilies.tsv"),
        pathabundance = join(PROJECT_DIR,  "humann3", "{sample}_pathabundance.tsv"),
        pathcoverage = join(PROJECT_DIR,  "humann3", "{sample}_pathcoverage.tsv"),
        taxon = join(PROJECT_DIR,  "humann3", "{sample}_metaphlan_bugs_list.tsv")
    params:
        outdir = join(PROJECT_DIR,  "humann3"),
        temp_taxon = join(PROJECT_DIR,  "humann3", "{sample}_humann_temp","{sample}_metaphlan_bugs_list.tsv")
    conda: "/data/home/chenliang/apps/miniconda3/envs/humann3_env"
    threads: 8
    shell:"""
        mkdir -p {params.outdir}

        humann3 --input {input.clean_read} \
        --output  {params.outdir} \
        --threads {threads} \
        --verbose \
        --output-basename {wildcards.sample} \
        --metaphlan-options "--bowtie2db /data/home/chenliang/DataBase/metaphlan4 --index mpa_vJun23_CHOCOPhlAnSGB_202403"

        cp {params.temp_taxon} {params.outdir}
    """

################################################################################
rule merge:
    input:
        taxon = expand(join(PROJECT_DIR, "humann3", "{sample}_metaphlan_bugs_list.tsv"), sample=sample_names),
        pathabundance = expand(join(PROJECT_DIR, "humann3", "{sample}_pathabundance.tsv"), sample=sample_names)
    output:
        res_taxon = join(PROJECT_DIR, "taxonomy.tsv"),
        res_path = join(PROJECT_DIR, "pathabundance.tsv"),
        res_rely_path = join(PROJECT_DIR, "pathabundance_relab.tsv"),
    conda: "/data/home/chenliang/apps/miniconda3/envs/humann3_env"
    params:
        inputdir  = join(PROJECT_DIR,  "humann3"),
        outdir = join(PROJECT_DIR)
    shell:"""
        merge_metaphlan_tables.py {input.taxon} | \
        sed 's/_metaphlan_bugs_list//g' | tail -n+2 | sed '1 s/clade_name/ID/' | sed '2i #metaphlan4'> {output.res_taxon}

        humann_join_tables --input {params.inputdir} \
        --file_name pathabundance \
        --output {output.res_path}

        humann_renorm_table \
        --input {output.res_path} \
        --units relab \
        --output {output.res_rely_path}

        humann_split_stratified_table \
        --input {output.res_rely_path} \
        --output {params.outdir}
    """

################################################################################
rule clean:
    input:
        rules.merge.output.res_taxon
    output: join(PROJECT_DIR, "cleaned")
    params:
        rmdir_1 = join(PROJECT_DIR,  "temp"),
        rmdir_2 = join(PROJECT_DIR,  "humann3","*_humann_temp"),
    shell: """
        rm -f {params.rmdir_1}/*.fq.gz
        rm -rf {params.rmdir_2}
        touch {output}
    """