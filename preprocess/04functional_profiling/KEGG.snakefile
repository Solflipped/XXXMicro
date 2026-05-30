from os.path import join, abspath, expanduser
import sys
import snakemake
import time

# 指定项目目录 specify project directories
PROJECT_DIR = config["output_directory"]
# 将PROJECT_DIR转换成绝对路径 convert PROJECT_DIR to absolute path
if PROJECT_DIR[0] == '~':
    PROJECT_DIR = expanduser(PROJECT_DIR)
PROJECT_DIR = abspath(PROJECT_DIR)


# 读取样本表（contigs_list.txt）
def get_sample_contigs(sample_file):
    sample_contigs = {}
    with open(sample_file) as lines:
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):   # 跳过空行和注释/标题行
                continue
            sample = line.split("\t")[0]
            contig = line.strip().split("\t")[1]
            sample_contigs[sample] = contig
    return sample_contigs

sample_contigs = get_sample_contigs(config['sample_contigs'])
sample_names = list(sample_contigs.keys())
print("top 10 sample_names: {}".format(sample_names[0:10]))
print("sample numbers: {}".format(len(sample_names)))



# 读取样本表（classification_input.txt）
def get_sample_reads(sample_file):
    sample_reads = {}
    paired_end = ''
    with open(sample_file) as sf:
        for l in sf.readlines():
            s = l.strip().split("\t")
            if len(s) == 1 or s[0] == 'Sample' or s[0] == '#Sample' or s[0].startswith('#'):
                continue
            sample = s[0]
            # paired end specified
            if (len(s)==3):
                reads = [s[1],s[2]]
                if paired_end != '' and not paired_end:
                    sys.exit('All samples must be paired or single ended.')
                paired_end = True
            # single end specified
            elif len(s)==2:
                reads=s[1]
                if paired_end != '' and paired_end:
                    sys.exit('All samples must be paired or single ended.')
                paired_end = False
            if sample in sample_reads:
                raise ValueError("Non-unique sample encountered!")
            sample_reads[sample] = reads
    return (sample_reads, paired_end)

# read in sample info and reads from the sample_file
sample_reads, paired_end = get_sample_reads(config['sample_reads'])


# 流程： 
# 1、先对contig进行过滤，然后使用prodigal进行基因预测
# 2、cd-hit进行去冗余，生成代表性基因集
# 3、功能注释

################################################################################
rule all:
    input:
        # 基因序列.fna
        genes = expand(join(PROJECT_DIR,"genes","{sample}_gene.fna"), sample=sample_names),
        # 蛋白序列.faa
        proteins = expand(join(PROJECT_DIR,"genes","{sample}_protein.faa"), sample=sample_names),
        # 代表性基因集
        cdhit_rep_seq = join(PROJECT_DIR, "cdhit_rep_seq.fna"),
        # 代表性基因集在 EggNOG-mapper 中的功能注释结果
        annotations= join(PROJECT_DIR,"cdhit_rep_seq.emapper.annotations"),
        # 基于 reads 比对和 CoverM 计算得到的每个样本的基因丰度表
        gene_abundance= expand(join(PROJECT_DIR,"coverm","{sample}_gene_abundance.txt"), sample=sample_names),
        # 结果
        merge_GOs= join(PROJECT_DIR,"merge_GOs.txt"),
        merge_KOs=join(PROJECT_DIR,"merge_KOs.txt"),
        merge_pathways=join(PROJECT_DIR,"merge_pathways.txt"),
        merge_modules = join(PROJECT_DIR, "merge_modules.txt")

################################################################################
rule filter_500bp:  # 过滤掉长度 <500bp 的 contigs
    input:
        contigs = lambda wildcards: sample_contigs[wildcards.sample]
    output:
        contigs_500bp = join(PROJECT_DIR,"contigs_500bp","{sample}_contigs.fa")
    params:
        sample = lambda wildcards: wildcards.sample
    conda: "/data/home/chenliang/apps/miniconda3"
    shell:"""
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/filter_contigs_500bp.py \
        {params.sample} {input.contigs} 500 {output.contigs_500bp}
    """

################################################################################
rule prodigal:  # 基因预测
    input:
        contigs_500bp = join(PROJECT_DIR,"contigs_500bp","{sample}_contigs.fa")
    output:
        gene = join(PROJECT_DIR,"genes","{sample}_gene.fna"),
        protein = join(PROJECT_DIR, "genes", "{sample}_protein.faa"),
        gff = join(PROJECT_DIR, "genes", "{sample}_gene.gff"),
        stat = join(PROJECT_DIR, "genes", "{sample}_stat.txt")
    conda: "/data/home/chenliang/apps/miniconda3/envs/prodigal_env"
    shell:"""
        prodigal -i {input.contigs_500bp} \
        -d {output.gene} -a {output.protein} -o {output.gff} \
        -f gff -p meta -s {output.stat} -q
    """

################################################################################
rule merge_gene_fna: # 合并所有样本的基因序列
    input:
        genes = expand(join(PROJECT_DIR,"genes","{sample}_gene.fna"), sample=sample_names)
    output:
        merge_genes = join(PROJECT_DIR,"merge_gene.fna")
    params:
        gene_dir=join(PROJECT_DIR,"genes")
    shell:"""
        cat {params.gene_dir}/*_gene.fna > {output.merge_genes}
    """

################################################################################
##reference: https://github.com/UriNeri/RVMT/blob/main/Clustering/DoubleClustering.sh#L20
rule linclust:  # 快速初筛、快速压缩
    input:
        merge_genes = join(PROJECT_DIR,"merge_gene.fna")
    output:
        linclust_genes = join(PROJECT_DIR, "linclust_rep_seq.fasta")
    params:
        out_prefix = join(PROJECT_DIR,"linclust"),
        out_temp = join(PROJECT_DIR,"linclust_temp")
    threads: 32
    conda: "/data/home/chenliang/apps/miniconda3/envs/mmseqs2_env"
    shell:"""
        mmseqs easy-linclust \
        --min-seq-id 0.95 -c 0.9 --cov-mode 1 --threads {threads} --split-memory-limit 16G \
        {input.merge_genes} {params.out_prefix} {params.out_temp} 
    """

################################################################################
##https://github.com/jiaonall/CRC-multi-kingdom/blob/main/1_raw%20sequence%20process.sh
rule cdhit:  # 精细聚类、生成最终catalog
    input: 
        linclust_genes = join(PROJECT_DIR, "linclust_rep_seq.fasta")
    output:
        representative_genes = join(PROJECT_DIR, "  a")
    threads: 32
    conda: "/data/home/chenliang/apps/miniconda3/envs/cd_hit_env"
    shell:"""
        cd-hit-est \
        -i {input.linclust_genes} -o {output.representative_genes} \
        -aS 0.9 -c 0.95 -G 0 -g 0 -T {threads} -M 0
    """

################################################################################
##https://github.com/eggnogdb/eggnog-mapper/wiki/eggNOG-mapper-v2
rule EggNOG_mapper_search:    # 基因功能注释 —— 去冗余后的代表性基因序列用于做同源基因比对（寻找同源基因）
    input:
        representative_genes = join(PROJECT_DIR,"cdhit_rep_seq.fna"),
        EggNOGdb = config['EggNOGdb']
    output:
        hits = join(PROJECT_DIR,"cdhit_rep_seq.emapper.hits"),
        seed_orthologs = join(PROJECT_DIR,"cdhit_rep_seq.emapper.seed_orthologs")
    params:
        out_prefix = join(PROJECT_DIR,"cdhit_rep_seq")
    conda: "/data/home/chenliang/apps/miniconda3/envs/eggnog_mapper_env"
    threads: 32
    shell:"""
        emapper.py -m diamond --no_annot --no_file_comments --itype CDS --cpu {threads} \
        --data_dir {input.EggNOGdb} -i {input.representative_genes} -o {params.out_prefix}
    """

################################################################################
rule EggNOG_mapper_annotate:   # 基因功能注释 —— 查询EggNOG数据库，得到功能注释
    input:
        seed_orthologs = join(PROJECT_DIR,"cdhit_rep_seq.emapper.seed_orthologs"),
        EggNOGdb= config['EggNOGdb']
    output:
        annotations = join(PROJECT_DIR,"cdhit_rep_seq.emapper.annotations")
    params:
        out_prefix = join(PROJECT_DIR,"cdhit_rep_seq")
    conda: "/data/home/chenliang/apps/miniconda3/envs/eggnog_mapper_env"
    threads: 32
    shell:"""
        emapper.py --annotate_hits_table {input.seed_orthologs}  \
        --no_file_comments --dbmem --cpu {threads} \
        --data_dir {input.EggNOGdb} \
        -o {params.out_prefix}
    """

################################################################################
##reference: https://github.com/deng-lab/viroprofiler/blob/main/modules/local/abundance.nf
rule mapping_reads_to_contigs:  # 把原始测序读段映射回去冗余基因集，生成比对的 BAM 文件
    input:
        r1 = lambda wildcards: sample_reads[wildcards.sample][0],
        r2 = lambda wildcards: sample_reads[wildcards.sample][1],
        representative_genes = join(PROJECT_DIR,"cdhit_rep_seq.fna")
    output:
        bam = join(PROJECT_DIR, "bam", "{sample}.bam")
    params:
        tempdir = join(PROJECT_DIR, "bam", "{sample}_temp_bam")
    conda: "/data/home/chenliang/apps/miniconda3/envs/coverm_env"
    threads: 4
    shell:"""
        # minimap2用于比对工具（把原始测序和去冗余后的代表性基因集进行比对） samtools用于排序（对minimap2产生对输出文件进行排序压缩）
        minimap2 -t {threads} \
        -ax sr --split-prefix {params.tempdir} {input.representative_genes} {input.r1} {input.r2} | \
        samtools sort --threads {threads} -o {output.bam}
    """

################################################################################
rule CoverM:  # 用 CoverM 统计每个基因的丰度
    input:
        bam = join(PROJECT_DIR, "bam", "{sample}.bam")
    output:
        gene_abundance = join(PROJECT_DIR, "coverm", "{sample}_gene_abundance.txt")
    params:
        temp_dir = join(PROJECT_DIR, "coverm", "{sample}_coverm_temp")
    threads: 4
    conda: "/data/home/chenliang/apps/miniconda3/envs/coverm_env"
    shell: """
        # coverm contig 模式，计算 contig（在这里是基因）覆盖度
        coverm contig \
        --bam-files {input.bam} -t {threads} --min-read-percent-identity 0.95 \
        --output-file {output.gene_abundance}
    """

################################################################################
##reference: https://github.com/NBISweden/nbis-meta/blob/main/workflow/scripts/eggnog-parser.py
rule eggNOG_parse:  # 把 EggNOG 注释解析成功能分组（GO、KO、Pathway、Module 等）
    input:
        KEGGdb = config['KEGGdb'],
        annotations = join(PROJECT_DIR,"cdhit_rep_seq.emapper.annotations")
    output:
        EggNOG_cazy = join(PROJECT_DIR, 'EggNOGdb', "cazy.parsed.tsv"),
        EggNOG_enzymes= join(PROJECT_DIR,'EggNOGdb',"enzymes.parsed.tsv"),
        EggNOG_gos= join(PROJECT_DIR,'EggNOGdb',"gos.parsed.tsv"),
        EggNOG_kos= join(PROJECT_DIR,'EggNOGdb',"kos.parsed.tsv"),
        EggNOG_modules= join(PROJECT_DIR,'EggNOGdb',"modules.parsed.tsv"),
        EggNOG_pathways= join(PROJECT_DIR,'EggNOGdb',"pathways.parsed.tsv"),
        EggNOG_tc= join(PROJECT_DIR,'EggNOGdb',"tc.parsed.tsv")
    params:
        EggNOGdb_dir = join(PROJECT_DIR,"EggNOGdb")
    shell:
        """
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/eggnog-parser.py parse \
        --map_go {input.KEGGdb} {input.annotations} {params.EggNOGdb_dir}
        """

################################################################################
rule eggNOG_quantify:  # 把基因丰度表映射到功能分组上，生成每个样本的功能丰度表
    input:
        EggNOG_gos = join(PROJECT_DIR,'EggNOGdb',"gos.parsed.tsv"),
        EggNOG_kos = join(PROJECT_DIR,'EggNOGdb',"kos.parsed.tsv"),
        EggNOG_pathways = join(PROJECT_DIR,'EggNOGdb',"pathways.parsed.tsv"),
        EggNOG_modules= join(PROJECT_DIR,'EggNOGdb',"pathways.parsed.tsv"),
        gene_abundance = join(PROJECT_DIR,"coverm", "{sample}_gene_abundance.txt")
    output:
        GOs = join(PROJECT_DIR,"functional_annotation", "{sample}_GOs.txt"),
        KOs = join(PROJECT_DIR,"functional_annotation", "{sample}_KOs.txt"),
        pathways = join(PROJECT_DIR,"functional_annotation", "{sample}_pathways.txt"),
        modules = join(PROJECT_DIR,"functional_annotation", "{sample}_modules.txt")
    shell:
        """
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/eggnog-parser.py quantify \
        {input.gene_abundance} {input.EggNOG_gos} {output.GOs}
        
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/eggnog-parser.py quantify \
        {input.gene_abundance} {input.EggNOG_kos} {output.KOs}
        
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/eggnog-parser.py quantify \
        {input.gene_abundance} {input.EggNOG_pathways} {output.pathways}
        
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/eggnog-parser.py quantify \
        {input.gene_abundance} {input.EggNOG_modules} {output.modules}
        """

################################################################################
rule eggNOG_merge: # 合并所有样本的功能丰度结果，得到矩阵
    input:
        GOs = expand(join(PROJECT_DIR,"functional_annotation","{sample}_GOs.txt"), sample=sample_names),
        KOs = expand(join(PROJECT_DIR,"functional_annotation","{sample}_KOs.txt"), sample=sample_names),
        pathways = expand(join(PROJECT_DIR,"functional_annotation","{sample}_pathways.txt"), sample=sample_names),
        modules= expand(join(PROJECT_DIR,"functional_annotation","{sample}_modules.txt"),sample=sample_names)
    output:
        merge_GOs = join(PROJECT_DIR,"merge_GOs.txt"),
        merge_KOs= join(PROJECT_DIR,"merge_KOs.txt"),
        merge_pathways= join(PROJECT_DIR,"merge_pathways.txt"),
        merge_modules= join(PROJECT_DIR,"merge_modules.txt")
    params:
        functional_annotation = join(PROJECT_DIR,"functional_annotation")
    shell:
        """
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/eggnog-parser.py merge \
        {params.functional_annotation}/*_GOs.txt {output.merge_GOs}
        
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/eggnog-parser.py merge \
        {params.functional_annotation}/*_KOs.txt {output.merge_KOs}
        
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/eggnog-parser.py merge \
        {params.functional_annotation}/*_pathways.txt {output.merge_pathways}
        
        python /data/home/chenliang/project/PRJCA022804/functional_profiling/scripts/eggnog-parser.py merge \
        {params.functional_annotation}/*_modules.txt {output.merge_modules}
        """



