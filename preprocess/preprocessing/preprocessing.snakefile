import re,os,subprocess
from os.path import join, expanduser, abspath

################################################################################
# 指定项目目录 specify project directories
DATA_DIR    = config["raw_reads_directory"]  # 获得原始测序读段段目录
PROJECT_DIR = config["output_directory"]     # 预处理项目流程存放地址
DataBase_DIR = config["database_directory"]  # 各分析工具所需用到的数据库目录
READ_SUFFIX = config["read_specification"]   # 正向or反向读段
EXTENSION   = config["extension"]            # 测序文件扩展名
# if gzipped, set this. otherwise not
gz_ext = '.gz' if EXTENSION.endswith('.gz') else ''
print(gz_ext)

# 将DATA_DIR、PROJECT_DIR和DataBase_DIR转换成绝对路径 convert PROJECT_DIR 、 DATA_DIR and DataBase_DIR to absolute path
if PROJECT_DIR[0] == '~':
    PROJECT_DIR = expanduser(PROJECT_DIR) # 将“～”代表的家目录转换成实际家目录路径
PROJECT_DIR = abspath(PROJECT_DIR) # 返回一个目录的绝对路径
if DATA_DIR[0] == '~':
    DATA_DIR = expanduser(DATA_DIR)
DATA_DIR = abspath(DATA_DIR)
if DataBase_DIR[0] == '~':
    DataBase_DIR = expanduser(DataBase_DIR)
DataBase_DIR = abspath(DataBase_DIR)

# 获取原始测序读段的文件名 get file names
FILES = []          # 所有 fastq 文件路径
SAMPLE_PREFIX = []  # 样本前缀
for root,dirs,files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(EXTENSION):
            sample = re.split('|'.join(['_' + a + '\.' for a in READ_SUFFIX]), file)[0]
            SAMPLE_PREFIX.append(sample)
            file = os.path.join(root,file)
            FILES.append(file)

print("files: ", FILES)
SAMPLE_PREFIX = list(set(SAMPLE_PREFIX)) # 去重
print("sample_prefix: ", SAMPLE_PREFIX)

# 流程： 
# 1、fastqc和multiqc生成质检报告 （使用conda安装环境）
# 2、hts_SuperDeduper去重 （使用conda安装htstream分析工具）
# 3、trim_galore质控、去宿主 （使用conda安装环境）
# 4、把每个样本最终生成的干净 reads 列表写成下游工具可直接读取的清单文件

################################################################################
rule all:
    input:
        # 质控报告
        expand(join(PROJECT_DIR, "00_qc_reports/pre_fastqc/{sample}_{read}_fastqc.html"), sample=SAMPLE_PREFIX, read=READ_SUFFIX),
        join(PROJECT_DIR, "00_qc_reports/pre_multiqc/multiqc_report.html"),
        expand(join(PROJECT_DIR, "00_qc_reports/post_fastqc/{sample}_{read}_fastqc.html"), sample=SAMPLE_PREFIX, read=['1', '2', 'orphans']),
        join(PROJECT_DIR, "00_qc_reports/post_multiqc/multiqc_report.html"),
        # 去宿主后文件
        expand(join(PROJECT_DIR, "04_sync/{sample}_1.fq.gz"), sample=SAMPLE_PREFIX),
        expand(join(PROJECT_DIR, "04_sync/{sample}_2.fq.gz"), sample=SAMPLE_PREFIX),
        expand(join(PROJECT_DIR, "04_sync/{sample}_orphans.fq.gz"), sample=SAMPLE_PREFIX),
        # 后续分类、组装所需要的样本清单
        join(PROJECT_DIR, "assembly_input.txt"),
        join(PROJECT_DIR, "classification_input.txt"),
        # 确保清除中间临时大文件
        join(PROJECT_DIR, "cleaned")


################################################################################
rule pre_fastqc:
    input:  
        fwd = join(DATA_DIR, "{sample}_" + READ_SUFFIX[0] + EXTENSION),
        rev = join(DATA_DIR, "{sample}_" + READ_SUFFIX[1] + EXTENSION)
    output:
        fwd = join(PROJECT_DIR,  "00_qc_reports/pre_fastqc/{sample}_" + READ_SUFFIX[0] + "_fastqc.html"),
        rev = join(PROJECT_DIR,  "00_qc_reports/pre_fastqc/{sample}_" + READ_SUFFIX[1] + "_fastqc.html")
    params:
        outdir = join(PROJECT_DIR, "00_qc_reports/pre_fastqc/")
    threads: min(4, len(READ_SUFFIX))
    conda: "/data/home/chenliang/apps/miniconda3/envs/fastqc_env"
    benchmark: join(PROJECT_DIR,  "00_qc_reports/pre_fastqc/{sample}_time.txt")
    shell: """
        mkdir -p {params.outdir}
        fastqc {input} --outdir {params.outdir} --threads {threads}
    """ 

rule pre_multiqc:
    input:  
        expand(join(PROJECT_DIR,  "00_qc_reports/pre_fastqc/{sample}_{read}_fastqc.html"), sample=SAMPLE_PREFIX, read=READ_SUFFIX)
    output:
        join(PROJECT_DIR,  "00_qc_reports/pre_multiqc/multiqc_report.html")
    params:
        indir = join(PROJECT_DIR,  "00_qc_reports/pre_fastqc"),
        outdir = join(PROJECT_DIR, "00_qc_reports/pre_multiqc/")
    conda: "/data/home/chenliang/apps/miniconda3/envs/multiqc_env"
    shell: """
        mkdir -p {params.outdir}
        multiqc --force {params.indir} -o {params.outdir}
    """ 
################################################################################
rule deduplicate:
    input:
        fwd = join(DATA_DIR, "{sample}_" + READ_SUFFIX[0] + EXTENSION),
        rev = join(DATA_DIR, "{sample}_" + READ_SUFFIX[1] + EXTENSION)
    output:
        fwd = join(PROJECT_DIR, "01_dedup/{sample}_1.fq.gz"),
        rev = join(PROJECT_DIR, "01_dedup/{sample}_2.fq.gz")
    params:
        outdir = join(PROJECT_DIR, "01_dedup/"),
        prefix = "{sample}"  # 用于hts_SuperDeduper的输出前缀
    threads: 1
    conda: "/data/home/chenliang/apps/miniconda3/envs/htstream_env"
    benchmark: join(PROJECT_DIR,  "01_dedup/{sample}_time.txt")
    shell: """
        mkdir -p {params.outdir} 
        # 使用绝对路径运行hts_SuperDeduper执行去重操作（这个去重仅针对PCR重复），并指定输出前缀
        hts_SuperDeduper -1 {input.fwd} -2 {input.rev} -f {params.outdir}{params.prefix} -F
        # 重命名输出文件以匹配规则输出
        mv {params.outdir}{params.prefix}_R1.fastq.gz {output.fwd}
        mv {params.outdir}{params.prefix}_R2.fastq.gz {output.rev}
    """ 
################################################################################
rule trim_galore:
    input:
        fwd = rules.deduplicate.output.fwd,
        rev = rules.deduplicate.output.rev
    output:
        fwd = join(PROJECT_DIR, "02_trimmed/{sample}_1_val_1.fq.gz"),
        rev = join(PROJECT_DIR, "02_trimmed/{sample}_2_val_2.fq.gz"),
        orp = join(PROJECT_DIR, "02_trimmed/{sample}_unpaired.fq.gz")
    threads: 4
    params:
        orp_fwd = join(PROJECT_DIR, "02_trimmed/{sample}_1_unpaired_1.fq.gz"),
        orp_rev = join(PROJECT_DIR, "02_trimmed/{sample}_2_unpaired_2.fq.gz"),
        q_min   = config['trim_galore']['quality'],
        min_len = config['trim_galore']['min_read_length'],
        outdir  = join(PROJECT_DIR, "02_trimmed/"),
    conda: "/data/home/chenliang/apps/miniconda3/envs/trim_galore_env"
    benchmark: join(PROJECT_DIR, "02_trimmed/{sample}_time.txt")
    shell: """
        mkdir -p {params.outdir}
        # 使用trim_galore丢弃接头和低质量读段
        trim_galore \
        --path_to_cutadapt /data/home/chenliang/apps/miniconda3/envs/trim_galore_env/bin/cutadapt \
        --quality {params.q_min} \
            --length {params.min_len} \
            --output_dir {params.outdir} \
            --paired {input.fwd} {input.rev} \
            --retain_unpaired \
            --cores {threads} 
        
        # 合并未配对读段并且压缩  merge unpaired and gzip
        zcat -f {params.orp_fwd} {params.orp_rev} | pigz -b 32 -p {threads} > {output.orp}
        # 删除中间文件  delete intermediate files
        rm {params.orp_fwd} {params.orp_rev}
    """
################################################################################
rule rm_host_reads:
    input:
        fwd = rules.trim_galore.output.fwd,
        rev = rules.trim_galore.output.rev,
        orp = rules.trim_galore.output.orp
    output:
        # 配对的双端读段中没有比对到宿主的序列 
        unmapped_1 = join(PROJECT_DIR, "04_sync/{sample}_1.fq.gz"),
        unmapped_2 = join(PROJECT_DIR, "04_sync/{sample}_2.fq.gz"),
        # 原本是未配对单端读段 + 配对双端读段中被“拆开的单端读段”，经过去宿主后剩下的序列
        unmapped_singletons = join(PROJECT_DIR, "04_sync/{sample}_orphans.fq.gz")
    params:
        outdir1  = join(PROJECT_DIR, "03_host_align/"),
        outdir2  = join(PROJECT_DIR, "04_sync/"),
        # BWA索引路径
        bwa_index_base = join(DataBase_DIR, "hg38/hg38.fa"),
        # 临时文件，用于存放单端读段，后面会合并到最终输出的unmapped_singletons
        singelton_temp_1 = join(PROJECT_DIR, "03_host_align/{sample}_rmHost_singletons1.fq.gz"),
        singelton_temp_2 = join(PROJECT_DIR, "03_host_align/{sample}_rmHost_singletons2.fq.gz")
    threads: 16
    conda: "/data/home/chenliang/apps/miniconda3/envs/align"
    benchmark: join(PROJECT_DIR, "03_host_align/{sample}_time.txt")
    shell: """
        mkdir -p {params.outdir1}
        mkdir -p {params.outdir2}
        # if an index needs to be built, use bwa index ref.fa
        # 在双端读段上运行 run on paired reads
        # bwa mem：将读段比对到宿主基因组
        # samtools fastq -f 4：只输出未比对到宿主的序列
        # -1/-2：未比对的双端读段输出到对应文件
        # -s：配对中拆开的单端读段输出到临时文件
        bwa mem -t {threads} {params.bwa_index_base} {input.fwd} {input.rev} | \
            samtools fastq -@ {threads} -t -T BX -f 4 -1 {output.unmapped_1} -2 {output.unmapped_2} -s {params.singelton_temp_1} -
        # 在单端读段（孤儿读段）上运行 run on unpaired reads
        # 单端读段直接比对宿主，未比对输出。
        bwa mem -t {threads} {params.bwa_index_base} {input.orp} | \
            samtools fastq -@ {threads} -t -T BX -f 4 - > {params.singelton_temp_2}
        # 合并单端读段
        zcat -f {params.singelton_temp_1} {params.singelton_temp_2} | pigz -p {threads} > {output.unmapped_singletons}
        rm {params.singelton_temp_1} {params.singelton_temp_2}
    """

################################################################################
rule post_fastqc:
    input:  join(PROJECT_DIR, "04_sync/{sample}_{read}.fq.gz")
    output: join(PROJECT_DIR,  "00_qc_reports/post_fastqc/{sample}_{read}_fastqc.html")
    params:
        outdir = join(PROJECT_DIR, "00_qc_reports/post_fastqc/")
    threads: 4
    conda: "/data/home/chenliang/apps/miniconda3/envs/fastqc_env"
    benchmark: join(PROJECT_DIR, "00_qc_reports/post_fastqc/{sample}_{read}_time.txt")
    shell: """
        mkdir -p {params.outdir}
        fastqc {input} -f fastq --outdir {params.outdir} -t {threads}
    """

rule post_multiqc:
    input: expand(join(PROJECT_DIR,  "00_qc_reports/post_fastqc/{sample}_{read}_fastqc.html"), sample=SAMPLE_PREFIX, read=['1', '2', 'orphans'])
    output: join(PROJECT_DIR,  "00_qc_reports/post_multiqc/multiqc_report.html")
    params:
        indir = join(PROJECT_DIR,  "00_qc_reports/post_fastqc"),
        outdir = join(PROJECT_DIR,  "00_qc_reports/post_multiqc/")
    conda: "/data/home/chenliang/apps/miniconda3/envs/multiqc_env"
    shell: """
        mkdir -p {params.outdir}
        multiqc --force {params.indir} -o {params.outdir}
    """

################################################################################
rule assembly_meta_file:
    input: expand(join(PROJECT_DIR,  "04_sync/{sample}_{read}.fq.gz"), sample=SAMPLE_PREFIX, read=['1', '2', 'orphans'])
    output: join(PROJECT_DIR, "assembly_input.txt")
    run:
        outfile = str(output)
        if (os.path.exists(outfile)):
            os.remove(outfile)
        with open(outfile, 'w') as outf:
            outf.writelines(['# Sample\tr1\tr2\torphans\n'])
            for sample in SAMPLE_PREFIX:
                outline = [sample, '\t'.join([
                join(PROJECT_DIR, "04_sync/" + sample + "_1.fq.gz"),
                join(PROJECT_DIR, "04_sync/" + sample + "_2.fq.gz"),
                join(PROJECT_DIR, "04_sync/" + sample + "_orphans.fq.gz")])]
                outf.writelines('\t'.join(outline) + '\n')

################################################################################
rule classification_meta_file:
    input: expand(join(PROJECT_DIR,  "04_sync/{sample}_{read}.fq.gz"), sample=SAMPLE_PREFIX, read=['1', '2', 'orphans'])
    output: join(PROJECT_DIR, "classification_input.txt")
    run:
        outfile = str(output)
        if (os.path.exists(outfile)):
            os.remove(outfile)
        with open(outfile, 'w') as outf:
            outf.writelines(['# Sample\tr1\tr2\n'])
            for sample in SAMPLE_PREFIX:
                outline = [sample, '\t'.join([
                join(PROJECT_DIR, "04_sync/" + sample + "_1.fq.gz"),
                join(PROJECT_DIR, "04_sync/" + sample + "_2.fq.gz")])]
                outf.writelines('\t'.join(outline) + '\n')

################################################################################
rule cleanup:
    input: expand(join(PROJECT_DIR,  "04_sync/{sample}_{read}.fq.gz"), sample=SAMPLE_PREFIX, read=['1', '2', 'orphans'])
    output: join(PROJECT_DIR, "cleaned")
    params:
        rmdir_1 = join(PROJECT_DIR, '01_dedup'),
        rmdir_2 = join(PROJECT_DIR, '02_trimmed')
    shell: """
        rm -f {params.rmdir_1}/*.fq.gz
        rm -f {params.rmdir_2}/*.fq.gz
        touch {output}
    """